This zip contains:
  * LGM_template.ipynb: Notebook with the python template functions that follow
      the programming assignment.
  * data: data directory that contains:
      - data.mat: file containing the data for the assignment. It can be opened 
                  with function 'load_dataset' in file LGM_template. The data
                  includes: position of joints (20 x 3 x 2045) corresponding to
                  20 joints x 3 positions (x,y,z) x 2045 individuals; labels
                  2045 and person index (2045 vector) indicating what is the
                  person doing the action
      - validation_data.mat: Contains an example of NB model and LG model for
        the data included in the file. Useful to validate your implementation.
        You can load it as:

          import scipy.io

          dd = scipy.io.loadmat('data/validation_data.mat')
          dd['data_small'] # Input data
          dd['labels_small'] # Input labels
          dd['individuals_small'] # Input individual indexes
          dd['train_indexes'] # Instances used for training
          dd['test_indexes']  # Instances used for test
          dd['model_nb']      # NB model
          dd['model_lg']      # LG model
          dd['accur_nb']      # Accuracy of NB model on test instances
          dd['accur_lg']      # Accuracy of LG model on test instances

      - ejemplolineargausian.mat: contains an execution example of LGM giving 
                  the expected output for a given input. The file can be opened
                  using:

          import scipy.io

          dd = scipy.io.loadmat('data/ejemplolineargaussian.mat')
          dd = dd['ejemplo'][0]
          # The inputs are
          X = dd['inputX']
          Y = dd['inputY']
          # The expected outputs are
          betas = dd['outputBetas']
          sigma = dd['outputSigma']

