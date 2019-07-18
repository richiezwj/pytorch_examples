def init_and_train(
    embedding_model: WordEmbedding,
    train_loader: data.DataLoader,
    validation_loader: data.DataLoader,
    loss_function: nn.modules.loss._Loss,
    configuration: Dict,
):

    """
    Train

    Arguments:
        model: model
        train_loader: dataloader for training set
        validation_loader: dataloader for vaidation set
        loss_function: loss function used for training
        configuration: configuration used for initializing model and training

    Return:
        None
    """

    # show model info
    print(model)
    
    # move model to the device
    model = model.to(DEVICE)

    # get one optimizer
    optimizer = optim.Adam(model.parameters(), lr=configuration["lr"])

    # record time taken
    start_time = time.time()

    # record best validation loss and initialize patience count to do early stopping manually
    best_val_loss = math.inf
    patience_count = 0

    for epoch in range(configuration["epoch"]):

        # set to training mode
        model.train()

        total_loss = 0.0
        total_acc = 0.0
        total_example = 0.0


        # load batch from data loader
        for ibatch, batch in enumerate(train_loader):

            # unwrap results
            token_idxs, lengths, label_idxs = batch

            # get batch size
            batch_size = token_idxs.size(0)

            # clear stored gradient
            model.zero_grad()

            # get model result
            tag_scores = model(token_idxs, lengths, DEVICE)

            # calculate loss
            loss = loss_function(tag_scores, label_idxs)

            # doing back propagation
            loss.backward()
            optimizer.step()

            # record results
            total_loss += loss.item()
            total_acc += count_accurate(tag_scores, label_idxs)
            total_example += batch_size

            if ibatch % configuration["log_interval"] == 0 and ibatch > 0:
                cur_loss = total_loss / configuration["log_interval"]
                cur_acc = total_acc / total_example
                elapsed = time.time() - start_time
                print(
                    "Training performance: | epoch {:3d} | batches {:3} | ms/batch {:5.2f} | loss {:5.2f} | acc {:5.2f}".format(
                        epoch,
                        ibatch,
                        elapsed * 1000 / configuration["log_interval"],
                        cur_loss,
                        cur_acc,
                    )
                )
                total_loss = 0.0
                total_acc = 0.0
                total_example = 0.0
                start_time = time.time()

        # evaluate on validation set
        val_loss, val_acc, val_examples = evaluate(
            model, validation_loader, loss_function
        )

        # compare validation loss with best validation loss for early stop
        if val_loss < best_val_loss:
            patience_count = 0
            best_val_loss = val_loss
            torch.save(model, model_fname)
        else:
            patience_count += 1

        if patience_count >= configuration["early_stop_patience"]:
            print(
                "Early stop triggered as validation loss didn't go down after {} epochs".format(
                    configuration["early_stop_patience"]
                )
            )
            print("Best validation loss: {:5.2f}".format(best_val_loss))
            break

