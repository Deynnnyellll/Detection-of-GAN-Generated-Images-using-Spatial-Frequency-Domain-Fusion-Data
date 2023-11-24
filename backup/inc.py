# application of incremental learning
def adapt(images, model_file, clf_file): 
    try:
        real = []
        gan = []
        real_labels = []
        gan_labels = []
        sorted_images = []

        for image in images:
            if "real" in image:
                real.append(image)
                label = np.ones(1)
                real_labels.append(label)
            elif "gan" in image:
                gan.append(image)
                label = np.zeros(1)
                gan_labels.append(label)

        sorted_images.extend(real)
        sorted_images.extend(gan)
            
        # preprocessing
        preprocessed_img = [preprocessing(image) for image in sorted_images]

        # apply spatial frequency feature fusion to the preprocessed images
        fused_features = spatial_frequency_feature_fusion(preprocessed_img)

        feature_vector = [feature.flatten() for feature in fused_features]

        labels = np.vstack((real_labels, gan_labels))
        true_labels = labels.reshape(labels.shape[0])

        os.chdir("/Users/Danniel/Downloads/Model/Validate")

        model_file1 = os.path.basename(model_file)
        print(model_file1)

        # incremental learning of svm
        model = train(true_labels, feature_vector, f'-s 1 -c 1 -B 1 -i {model_file1}')

            # predict new value and get the svm scores to add in the platt scaler
        _, _, svm_scores = predict(true_labels, feature_vector, model)

        # move to the platt scaler dir to access the regression model
        os.chdir("/Users/Danniel/Detection-of-GAN-Generated-Images-using-Spatial-Frequency-Domain-Fusion-Data/platt scaler")
        clf = os.path.basename(clf_file)
        # incremental learning of platt scaler
        plat = train(true_labels, svm_scores, f'-s 0 -c 1 -B 1 -i {clf}')

        os.chdir("/Users/Danniel/Detection-of-GAN-Generated-Images-using-Spatial-Frequency-Domain-Fusion-Data")

        return model, plat
    except Exception as e:
        print(f"Incremental Learning Error: {e}")