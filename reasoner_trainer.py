def prepare_data(encoded_doc):
    # create training set
    train_size = int(0.7 * len(encoded_doc))
    trainDataloader = encoded_doc[:train_size]
    
    rest = len(encoded_doc) - train_size
    restLS = encoded_doc[train_size:]
    
    # create validation set 
    val_size = int(0.15 * rest)
    validationDataloader = restLS[:val_size]
    
    # create test set
    test_size = rest - val_size
    predictionDataloader = restLS[val_size:]

    return (trainDataloader, validationDataloader, predictionDataloader)
    