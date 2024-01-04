def classify_event(image):
    model = FiveClassifier(len(unique_labels)).to(DEVICE)
    model.load_state_dict(torch.load("your_model_weights.pth"))  # Load your trained model weights

    img = cv2.imread(image)
    img = cv2.resize(img, (256, 256))
    img = img.transpose((2, 0, 1))
    img = torch.tensor(img, dtype=torch.float).unsqueeze(0) / 255.0  # Add batch dimension and normalize

    with torch.no_grad():
        model.eval()
        img = img.to(DEVICE)
        y_pred = model(img)

    event_index = y_pred.argmax().item()
    event_names = unique_labels
    detected_event = event_names[event_index]

    return detected_event


