import torch
import json
import torchtext; torchtext.disable_torchtext_deprecation_warning()

from torch.utils.data import DataLoader

from data import get_training_data
from model import get_model

if __name__ == "__main__":

    model, vocab, device = get_model()

    labels, dataset = get_training_data("training_data.csv", vocab)
    dl = DataLoader(
            dataset, 
            batch_size=64, 
            shuffle=True, 
            collate_fn=dataset.collate_fn, 
            pin_memory=True
    )
    inputs, outputs = next(iter(dl))
    # export_options = torch.onnx.ExportOptions(fake_context=fake_context)

    model = model.to(device)
    inputs = inputs.to(device)

    onnx_program = torch.onnx.dynamo_export(
            model,
            inputs, 
            # export_options=export_options
    )
    onnx_program.save("classifier.onnx", model_state=torch.load("classifier.pth"))
