import argparse
import train
import test

#keras_models= ['inceptionv3']

if __name__ == '__main__':
    task = 'train'
    data_dir_train = "./data/train_data"
    data_dir_valid = "./data/validate_data"
    data_dir_test = "./data/test_data"
    model_name = "inceptionv3"
    epochs = 100
    batch_size = 16
    training_type = "train_all"
    save_loc = "models/"
    weights' = 'imagenet'
    results_loc = 'results/'

    #print_arguments()
    print('task : ', task)
    print('data_dir_train : ', data_dir_train)
    print('data_dir_valid : ', data_dir_valid)
    print('data_dir_test : ', data_dir_test)
    print('model_name : ', model_name)
    print('epochs : ', epochs)
    print('batch_size :', batch_size)
    print('training_type :', training_type)
    print('save_loc : ', save_loc)
    print('weights : ', weights)
    print('results_loc : ', results_loc)


    if task.lower() == 'train':
        output_string = train.train_model(data_dir_train, data_dir_valid, batch_size, epochs, model_name, training_type, save_loc, weights)
    elif args.task.lower() == 'test':
        output_string = test.test_model(data_dir_train, data_dir_test, batch_size, model_name, save_loc, results_loc)
    else:
        output_string = "Incorrect Task"
    print(output_string)

