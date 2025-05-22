from torch import nn
import torch

# Define a custom neural network module named 'attention_simi_guided_loss' which inherits from nn.Module
class attention_simi_guided_loss(nn.Module):
    def __init__(self, 
                 threshold=0.6, 
                 temperature=0.04, 
                 loss_type="BCEWithLogitsLoss"
                 ):
        super(attention_simi_guided_loss, self).__init__()
        self.temperature = temperature
        self.threshold = threshold
        self.attention_mse = nn.MSELoss()
        self.eps = 1e-6
        if loss_type == "BCEWithLogitsLoss":
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def threshold_attention_map(self, attention_map:torch.Tensor):
        # Sort the values in the attention map along the last dimension in descending order,
        # and also get the corresponding indices
        sorted_values, sorted_indices = torch.sort(attention_map, dim=-1, descending=True)
        # Calculate the cumulative sum of the sorted values along the last dimension
        cumulative_sum = torch.cumsum(sorted_values, dim=-1)
        # Create a boolean mask where cumulative sum is less than or equal to the threshold
        threshold_mask = cumulative_sum <= self.threshold
        # Ensure the mask is consistent for all elements after the first one in the last dimension
        threshold_mask[..., 1:] &= threshold_mask[..., :-1]
        # Create a tensor of zeros with the same shape as the attention map
        binary_map = torch.zeros_like(attention_map)
        # Convert the data type of the threshold mask to match that of the binary map
        threshold_mask = threshold_mask.to(binary_map.dtype)
        # Use the scatter_ function to fill the binary map with values from the threshold mask
        # based on the sorted indices
        binary_map.scatter_(-1, sorted_indices, threshold_mask)
        return binary_map

    # Calculate the multi-label loss
    def multi_label_loss(self, preds:torch.Tensor, attention_map:torch.Tensor, ir_info=None):
        """
            preds Tensor shape (batch, num_patches, num_patches)
            attention_map Tensor shape similar to preds
        """
        # If no additional information (ir_info) is provided, just use the original attention map
        if ir_info is None:
            attention_map = attention_map
        else:
            # Convert the attention map to the same data type as the predictions
            attention_map = attention_map.to(preds.dtype)
            # Flatten the 'ir_info' tensor to a 2D tensor
            ir_info_flattened = ir_info.view(attention_map.size(0), -1)
            # Create a mask where values in 'ir_info_flattened' are equal to 1, and convert it to float
            mask = (ir_info_flattened == 1).float()
            # Expand the mask to have an additional dimension at the end
            mask_expanded = mask.unsqueeze(-1)
            # Adjust the attention map based on the mask
            attention_map -= mask_expanded * 0.7
            # Clamp the attention map values to be at least 0
            attention_map = torch.clamp(attention_map, min=0)
        # Calculate the loss using the pre-defined criterion
        return self.criterion(preds, attention_map)

    # Implement the forward pass of the module
    def __call__(self, 
                 vis_embeds:torch.Tensor, 
                 ir_embeds:torch.Tensor, 
                 attention_map: torch.Tensor,
                 ir_info=None
                 ):
        attention_map = self.threshold_attention_map(attention_map.sum(dim=1) / attention_map.size(1)).cuda()
        vis_embeds = vis_embeds / (vis_embeds.norm(p=2, dim=-1, keepdim=True) + self.eps)
        ir_embeds = ir_embeds / (ir_embeds.norm(p=2, dim=-1, keepdim=True) + self.eps)
        logits = torch.bmm(ir_embeds, vis_embeds.transpose(2,1)) / self.temperature

        loss1  = self.multi_label_loss(logits, attention_map, ir_info=ir_info)
        loss2 = self.multi_label_loss(logits.transpose(2,1), attention_map.transpose(2,1), ir_info=ir_info)

        return (loss1 + loss2) / 2