
class medical_NeuroNER(object):
    def __init__(self, parameters, metadata):
        mode = parameters['mode']

        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=parameters['number_of_cpu_threads'],
            inter_op_parallelism_threads=parameters['number_of_cpu_threads'],
            device_count={'CPU': 1, 'GPU': parameters['number_of_gpus']},
            allow_soft_placement=True,
            log_device_placement=False
        )

        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.90
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            if mode == "train" or not parameters['use_vocab_expansion']:
                use_vocab_expansion = False
            else:
                use_vocab_expansion = True
            model = EntityLSTM(parameters, metadata, use_external_embedding=use_vocab_expansion)
            sess.run(tf.global_variables_initializer())

        if mode == "train":
            if parameters['load_pretrained_model']:
                self.transition_params_trained = model.load_model(parameters['pretrained_model_folder'], sess, metadata,
                                                                  parameters)
            else:
                model.load_pretrained_token_embeddings(sess, parameters, metadata)
                self.transition_params_trained = np.random.rand(metadata['num_of_label'] + 2,
                                                                metadata['num_of_label'] + 2)
        else:
            self.transition_params_trained = model.load_model(parameters['pretrained_model_folder'], sess, metadata,
                                                              parameters)

        expanded_embedding = None

        self.expanded_embedding = expanded_embedding
        self.model = model
        self.parameters = parameters
        self.sess = sess
        self.metadata = metadata

    @staticmethod
    def batch_extract_feature(input, parameters, gazetteer, max_key_len, metadata, expanded_embedding):
        token_sequence, raw_token_sequence, extended_sequence = preprocess.extract_feature(input,
                                                                                           parameters[
                                                                                               'tokenizer'],
                                                                                           gazetteer,
                                                                                           max_key_len)
        model_input = preprocess.encode(metadata, token_sequence, extended_sequence,
                                        expanded_embedding=expanded_embedding)
        return raw_token_sequence, extended_sequence, model_input, raw_token_sequence


    def fit(self, dataset_filepaths):
        stats_graph_folder, experiment_timestamp = self._create_stats_graph_folder(self.parameters)
        # Initialize and save execution details
        start_time = time.time()
        results = {}
        results['epoch'] = {}
        results['execution_details'] = {}
        results['execution_details']['train_start'] = start_time
        results['execution_details']['time_stamp'] = experiment_timestamp
        results['execution_details']['early_stop'] = False
        results['execution_details']['keyboard_interrupt'] = False
        results['execution_details']['num_epochs'] = 0
        results['model_options'] = copy.copy(self.parameters)

        model_folder = os.path.join(stats_graph_folder, 'model')
        utils.create_folder_if_not_exists(model_folder)
        del self.metadata['prev_num_of_token']
        del self.metadata['prev_num_of_char']

        copyfile(self.parameters['ini_path'], os.path.join(model_folder, 'parameters.ini'))

        if self.parameters['enable_tensorbord']:
            tensorboard_log_folder = os.path.join(stats_graph_folder, 'tensorboard_logs')
            utils.create_folder_if_not_exists(tensorboard_log_folder)
            tensorboard_log_folders = {}
            for dataset_type in dataset_filepaths.keys():
                tensorboard_log_folders[dataset_type] = os.path.join(stats_graph_folder, 'tensorboard_logs',
                                                                     dataset_type)
                utils.create_folder_if_not_exists(tensorboard_log_folders[dataset_type])

            # Instantiate the writers for TensorBoard
            writers = {}
            for dataset_type in dataset_filepaths.keys():
                writers[dataset_type] = tf.summary.FileWriter(tensorboard_log_folders[dataset_type],
                                                              graph=self.sess.graph)
            embedding_writer = tf.summary.FileWriter(
                model_folder)  # embedding_writer has to write in model_folder, otherwise TensorBoard won't be able to view embeddings

            projector.visualize_embeddings(embedding_writer, embeddings_projector_config)

            # Write metadata for TensorBoard embeddings
            token_list_file = open(token_list_file_path, 'w', encoding='UTF-8')
            for key, _ in self.metadata['token_to_index'].items():
                token_list_file.write('{0}\n'.format(key))
            token_list_file.close()

            if self.parameters['use_character_lstm']:
                character_list_file = open(character_list_file_path, 'w', encoding='UTF-8')
                for key, _ in self.metadata['character_to_index'].items():
                    character_list_file.write('{0}\n'.format(key))
                character_list_file.close()

        # Start training + evaluation loop. Each iteration corresponds to 1 epoch.
        bad_counter = 0  # number of epochs with no improvement on the validation test in terms of F1-score
        previous_best_valid_f1_score = 0

        data_queue = {}
        for dataset_type, dataset_path in dataset_filepaths.items():
            data_queue[dataset_type] = DataQueue(self.metadata, dataset_path, self.parameters['batch_size'],
                                                 is_train=True if dataset_type == 'train' else False,
                                                 # use_process=True if dataset_type == 'train' else False,
                                                 use_process=False,
                                                 pad_constant_size=self.parameters['use_attention'])

        first_step = True
        try:
            accum_step = 0
            step = 0
            while True:
                print('\nStarting step {0}'.format(accum_step))

                epoch_start_time = time.time()

                if not first_step:
                    bar = tqdm(total=BREAK_STEP)
                    while True:
                        if step > BREAK_STEP:
                            step %= BREAK_STEP
                            break
                        batch_input = data_queue['train'].next()
                        self.transition_params_trained = self._train_step(batch_input)
                        step += self.parameters['batch_size']
                        accum_step += self.parameters['batch_size']
                        # print('Training {0:.2f}% done'.format(step / BREAK_STEP * 100), end='\r', flush=True)
                        bar.update(self.parameters['batch_size'])
                    epoch_elapsed_training_time = time.time() - epoch_start_time
                    print('Training completed in {0:.2f} seconds'.format(epoch_elapsed_training_time), flush=True)
                    bar.close()
                first_step = False

                # Predict labels using trained model
                y_pred = {}
                y_true = {}
                output_filepaths = {}
                # for dataset_type in ['train', 'valid', 'test', 'deploy']:
                for dataset_type in ['valid', 'test', 'deploy']:
                    if dataset_type not in data_queue:
                        continue
                    prediction_output = self._prediction_step(data_queue, dataset_type,
                                                              stats_graph_folder, accum_step)
                    y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output

                # Evaluate model: save and plot results
                evaluate.evaluate_model(results, self.metadata, y_pred, y_true, stats_graph_folder, accum_step,
                                        epoch_start_time, output_filepaths, self.parameters)

                # Save model
                self.model.saver.save(self.sess, os.path.join(model_folder, 'model_{0:05d}.ckpt'.format(accum_step)))
                self.metadata.write(model_folder)

                if self.parameters['enable_tensorbord']:
                    # Save TensorBoard logs
                    summary = self.sess.run(self.model.summary_op, feed_dict=None)
                    writers['train'].add_summary(summary, accum_step)
                    writers['train'].flush()
                    utils.copytree(writers['train'].get_logdir(), model_folder)

                # Early stop
                valid_f1_score = results['epoch'][accum_step][0]['valid']['f1_score']['micro']
                if valid_f1_score > previous_best_valid_f1_score:
                    bad_counter = 0
                    previous_best_valid_f1_score = valid_f1_score
                    self.model.saver.save(self.sess, os.path.join(model_folder, 'model.ckpt'))
                    self.metadata.write(model_folder)
                else:
                    bad_counter += 1
                print("The last {0} epochs have not shown improvements on the validation set.".format(bad_counter))

                if bad_counter >= self.parameters['patience']:
                    print('Early Stop!')
                    results['execution_details']['early_stop'] = True
                    break

                if accum_step >= self.parameters['maximum_number_of_steps']: break


        except KeyboardInterrupt:
            results['execution_details']['keyboard_interrupt'] = True
            print('Training interrupted')

        print('Finishing the experiment')

        end_time = time.time()
        results['execution_details']['train_duration'] = end_time - start_time
        results['execution_details']['train_end'] = end_time
        evaluate.save_results(results, stats_graph_folder)
        if self.parameters['enable_tensorbord']:
            for dataset_type in dataset_filepaths.keys():
                writers[dataset_type].close()

    def _create_stats_graph_folder(self, parameters):
        # Initialize stats_graph_folder
        experiment_timestamp = utils.get_current_time_in_miliseconds()
        dataset_name = utils.get_basename_without_extension(parameters['dataset_text_folder'])
        model_name = '{0}_{1}'.format(dataset_name, experiment_timestamp)
        utils.create_folder_if_not_exists(parameters['output_folder'])
        stats_graph_folder = os.path.join(parameters['output_folder'], model_name)  # Folder where to save graphs
        utils.create_folder_if_not_exists(stats_graph_folder)
        return stats_graph_folder, experiment_timestamp

    def close(self):
        self.__del__()

    def __del__(self):
        self.sess.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter', type=str, default='./parameters.ini')
    parser.add_argument('--mode', type=str, default='')
    cmd_arg = parser.parse_args()

    start_time = time.time()

    print('Init... ', end='', flush=True)
    parameters = Configuration(cmd_arg.parameter)

    if cmd_arg.mode != '':
        parameters['mode'] = cmd_arg.mode
    dataset_filepaths = utils.get_valid_dataset_filepaths(parameters['dataset_text_folder'])

    if parameters['mode'] == 'train':
        dataset_filepaths.pop('test', None)
        if 'train' not in dataset_filepaths or 'valid' not in dataset_filepaths:
            raise Exception('train data path empty')

    elif parameters['mode'] == 'test':
        dataset_filepaths.pop('train', None)
        if len(dataset_filepaths) == 0:
            raise Exception('test data path empty')


    neuroner = medical_NeuroNER(parameters, metadata)
    print('done ({0:.2f} seconds)'.format(time.time() - start_time))

    if parameters['mode'] == 'train':
        neuroner.fit(dataset_filepaths)

    elif parameters['mode'] == 'test':
        neuroner.test(dataset_filepaths)

    else:
        raise Exception("error")
    neuroner.close()

    print('complete ({0:.2f} seconds)'.format(time.time() - start_time))


if __name__ == "__main__":
    main()
