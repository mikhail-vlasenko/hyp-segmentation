import torch
import lightning as L
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

from pascal_voc_dataset import create_pascal_voc_prototypes_from_spin, PASCAL_VOC_SUBSET_CLASSES


class PascalVOCValidationCallback(L.Callback):
    """
    Callback to evaluate the model on Pascal VOC dataset during training.
    This runs Pascal VOC evaluation separately from the main SPIN validation.
    """
    
    def __init__(self, pascal_voc_dataloader, device="auto"):
        super().__init__()
        self.pascal_voc_dataloader = pascal_voc_dataloader
        self.device = device
        
        # Initialize metrics for Pascal VOC subset (classes that match SPIN)
        self.num_classes = len(PASCAL_VOC_SUBSET_CLASSES)
        self.pascal_voc_miou = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=255,  # Standard Pascal VOC ignore index
        )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Run Pascal VOC evaluation at the end of each validation epoch."""
        pl_module.eval()
        
        self.pascal_voc_miou.reset()
        self.pascal_voc_miou.to(pl_module.device)

        whole_head = pl_module.decode_heads["whole"]
        
        # Save original prototypes
        original_prototypes = whole_head.prototypes["whole"]
        
        # Create Pascal VOC prototypes from SPIN prototypes
        pascal_prototypes = create_pascal_voc_prototypes_from_spin(original_prototypes)
        
        # Temporarily replace prototypes with Pascal VOC prototypes
        whole_head.prototypes["whole"] = pascal_prototypes
        
        with torch.no_grad():
            for batch in tqdm(self.pascal_voc_dataloader):
                # Move batch to device
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                
                # Forward pass with Pascal VOC prototypes
                logits, _ = pl_module.forward(batch["pixel_values"], eval_mode=True)
                
                # Use the "whole" granularity logits (now computed with Pascal prototypes)
                whole_logits = logits["whole"]
                
                # Upsample logits to match label size
                upsampled_logits = pl_module.upsample_logits(whole_logits, batch["labels"])
                
                # Get predictions
                preds = torch.argmax(upsampled_logits, dim=1)
                
                # Update metrics using the base labels
                labels = batch["labels"]
                
                # Update Pascal VOC mIoU
                self.pascal_voc_miou.update(preds, labels)

                if trainer.sanity_checking:
                    break
        
        whole_head.prototypes["whole"] = original_prototypes
        
        pascal_voc_miou_score = self.pascal_voc_miou.compute()
        
        if not trainer.sanity_checking:
            pl_module.log("val_pascal_voc_mIoU", pascal_voc_miou_score, prog_bar=True, sync_dist=True)

        self.pascal_voc_miou.reset()
        pl_module.train()
