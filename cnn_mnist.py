import tensorflow as tf

from models.cnn_mnist_model import CnnMnistModel
from data_loader.data_generator import DataGenerator
from trainers.cnn_mnist_trainer import CnnMnistTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    # use mnist dataset
    data.load_mnist()
    
    # create an instance of the cnn model
    model = CnnMnistModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = CnnMnistTrainer(sess, model, data, config, logger)
    #load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()



if __name__ == '__main__':
    main()
