from torch import nn
import torch
    
class RGB_patch_simi_loss(nn.Module):
    def __init__(self):
        super(RGB_patch_simi_loss, self).__init__()
        self.MSE = nn.MSELoss(reduction ='mean').cuda()
        self.eps = 1e-6

    def __call__(self, Teacher_vis_embeddings:torch.Tensor , Student_vis_embeddings:torch.Tensor):
        """
        args:
            Teacher_vis_embeddings: (bs , num_patches , embed_dim)
            Student_vis_embeddings: (bs , num_patches , embed_dim)
        """
        # Normalized features.
        Teacher_vis_embeddings = Teacher_vis_embeddings / (Teacher_vis_embeddings.norm(p=2, dim=-1, keepdim=True) + self.eps)
        Student_vis_embeddings = Student_vis_embeddings / (Student_vis_embeddings.norm(p=2, dim=-1, keepdim=True) + self.eps)

        # Calculating the Loss
        T_logits = torch.bmm(Teacher_vis_embeddings, Teacher_vis_embeddings.transpose(2,1))
        S_logits = torch.bmm(Student_vis_embeddings, Student_vis_embeddings.transpose(2,1))

        return self.MSE(T_logits, S_logits)
