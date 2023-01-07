import argparse
import json
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
import shutil
import anafora
import anafora.evaluate
from keras.utils import to_categorical
from transformers import BertTokenizerFast
from transformers import RobertaTokenizerFast, TFRobertaForTokenClassification, AutoConfig

# attempt to make results deterministic
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(42)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
# list of possible time expression labels
labels = [
    None,
    'AMPM-Of-Day',
    'After',
    'Before',
    'Between',
    'Calendar-Interval',
    'Day-Of-Month',
    'Day-Of-Week',
    'Frequency',
    'Hour-Of-Day',
    'Last',
    'Minute-Of-Hour',
    'Modifier',
    'Month-Of-Year',
    'Next',
    'NthFromStart',
    'Number',
    'Part-Of-Day',
    'Part-Of-Week',
    'Period',
    'Season-Of-Year',
    'Second-Of-Minute',
    'Sum',
    'This',
    'Time-Zone',
    'Two-Digit-Year',
    'Union',
    'Year',
]

# mapping from labels to integer indices
label_to_index = {l: i for i, l in enumerate(labels)}

seq_length = 64
batch_size = None

def train(model_dir, data_dir, epochs, ext_model_dir=None):
    # tokenizor
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    # read training data and extract tokens with labels
    train_features = []
    train_labels = []
    text_index = []

    for _, text, _, data in iter_data(data_dir, "gold"):
        span_types = {
            span: annotation.type
            for annotation in data.annotations
            for span in annotation.spans
        }

        # tokenize input text
        text_input = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)

        # re-tokenize label
        span_types_re_token = {}
        for start, end in span_types.keys():
            label_token_list = [item for item in text_input['offset_mapping'] if item[0] >= start and item[1] <= end]
            for start_2, end_2 in label_token_list:
                span_types_re_token.update({(start_2, end_2): span_types[(start, end)]})

        train_labels_temp = []
        for start, end in text_input['offset_mapping']:
            label = span_types_re_token.get((start, end))
            train_labels_temp.append(label_to_index[label])
        train_features_temp = text_input["input_ids"]

        # padding to n*seq_length
        slice_num = int(len(train_labels_temp) / seq_length) + 1
        train_labels_temp = train_labels_temp + [1] * (slice_num * seq_length - len(train_labels_temp))
        train_features_temp = train_features_temp + [1] * (slice_num * seq_length - len(train_features_temp))

        for i in range(slice_num):
            text_index.append(i)
            train_features.append(train_features_temp[i * seq_length:(i + 1) * seq_length])
            train_labels.append(train_labels_temp[i * seq_length:(i + 1) * seq_length])

    train_features = np.vstack(train_features)
    train_labels = np.vstack(train_labels)

    #############Create a Model
    #model parameters
    config = AutoConfig.from_pretrained(
        "roberta-base",
        num_labels=len(labels),
        # hidden_dropout_prob=0.2,
        # hidden_size=408, # a multiple of the number of attention heads (12)
        # attention_probs_dropout_prob=0.2,
        kernel_initializer="glorot_uniform",
        num_parameters=(False, False),
        # max_position_embeddings=512,
        # type_vocab_size=28,
        output_attentions=False,
        output_hidden_states=False,
        # train_mask=train_mask,
    )

    # train the model
    model = TFRobertaForTokenClassification.from_pretrained('roberta-base', config=config)

    if ext_model_dir != None:
        model = TFRobertaForTokenClassification.from_pretrained(ext_model_dir)

    model.compile(tf.keras.optimizers.Adam(learning_rate= 5e-6), loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.fit(train_features, train_labels, epochs=epochs, batch_size=batch_size, shuffle=True)


    # save the model
    model.save_pretrained(model_dir)


def predict(model_dir, output_dir, text_dir, reference_dir):
    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    # read the model and token index
    #model = tf.keras.models.load_model(model_dir)
    model = TFRobertaForTokenClassification.from_pretrained(model_dir)

    # write one file of predictions for each input text file
    for text_path, text, xml_path, data in iter_data(text_dir, "system"):
        text_input = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        # convert tokens to features
        dev_features_temp = text_input["input_ids"]
        dev_map_temp = text_input['offset_mapping']
        text_index = []
        dev_features = []
        dev_map = []
        slice_num = int(len(dev_features_temp) / seq_length) + 1
        dev_features_temp = dev_features_temp + [1] * (slice_num * seq_length - len(dev_features_temp))
        last_char_idx = dev_map_temp[-1][1]
        for i in range(slice_num * seq_length - len(dev_map_temp)):
            dev_map_temp += [(last_char_idx + (i + 1), last_char_idx + (i + 2))]

        for i in range(slice_num):
            text_index.append(i)
            dev_features.append(dev_features_temp[i * seq_length:(i + 1) * seq_length])
            dev_map.append(dev_map_temp[i * seq_length:(i + 1) * seq_length])
        dev_features = np.array(dev_features)

        predictions = np.argmax(model.predict(dev_features), axis=-1)
        predictions=predictions.reshape(predictions.shape[1:])

        # convert predictions into XML
        for j, slices in enumerate(dev_map):
            slices_predictions = predictions[j]
            for i, (start, end) in enumerate(slices):
                if slices_predictions[i] != 0:
                    if i == 0:
                        start_temp = start
                    elif slices_predictions[i] != slices_predictions[i - 1]:
                        start_temp = start

                    if i != len(slices_predictions) - 1:
                        if slices_predictions[i] == slices_predictions[i + 1]:
                            continue

                    entity = anafora.AnaforaEntity()
                    entity.id = f"{(j * seq_length + i)}@e@{text_path}@system"
                    entity.type = labels[slices_predictions[i]]
                    entity.spans = (start_temp, end),
                    data.annotations.append(entity)

        # write the XML file to the output directory
        xml_rel_path = os.path.relpath(xml_path, text_dir)
        output_path = os.path.join(output_dir, xml_rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_file(output_path)
    # package the predictions up for CodaLab
    shutil.make_archive(output_dir, "zip", output_dir)

    # if a reference directory is provided, evaluate the predictions
    f1_score = 0
    if reference_dir is not None:
        file_scores = anafora.evaluate.score_dirs(
            reference_dir, output_dir, exclude={("*", "<span>")})
        f1_score = anafora.evaluate._print_merged_scores(
            file_scores, anafora.evaluate.Scores)
    return f1_score


def iter_data(root_dir, xml_type):
    # walk down to each directory with online files (no subdirectories)
    for dir_path, dir_names, file_names in os.walk(root_dir):
        if not dir_names:

            # read the text from the text file
            [text_file_name] = [f for f in file_names if not f.endswith(".xml")]
            text_path = os.path.join(dir_path, text_file_name)
            with open(text_path) as text_file:
                text = text_file.read()

            # calculate the XML file name from the text file name
            xml_path = f"{text_path}.TimeNorm.{xml_type}.completed.xml"

            # read the gold annotations or create an empty annotations object
            if xml_type == 'gold':
                data = anafora.AnaforaData.from_file(xml_path)
            elif xml_type == 'system':
                data = anafora.AnaforaData()
            else:
                raise ValueError(f"unsupported xml_type: {xml_type}")

            # generate a tuple for this document
            yield text_path, text, xml_path, data


def copy_model(old_model_dir, new_model_dir):
    model = TFRobertaForTokenClassification.from_pretrained(old_model_dir)
    model.save_pretrained(new_model_dir)


if __name__ == "__main__":
    # sets up a command-line interface for "train", "predict" and "evaluate"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)
    train_parser.add_argument("model_dir", nargs='?', default='model')
    train_parser.add_argument("data_dir", nargs='?', default='train')
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--ext", dest='ext_model_dir') #get pre-train model
    predict_parser = subparsers.add_parser("predict")
    predict_parser.set_defaults(func=predict)
    predict_parser.add_argument("model_dir", nargs='?', default='model')
    predict_parser.add_argument('output_dir', nargs='?', default='submission')
    predict_parser.add_argument("text_dir", nargs='?', default='dev')
    predict_parser.add_argument("--evaluate", dest='reference_dir')
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop("command")
    kwargs.pop("func")(**kwargs)
