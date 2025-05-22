import torch 
from torch import nn

def load_model_backbone(model:nn.Module,pretrain_path,model_name):

    saved_model = torch.load(pretrain_path,map_location='cpu') 
    # for key, value in saved_model.items():
    #     print(f"{key}") 

    if any(model_name in key for key in saved_model.keys()):
        print(f"The saved model contains keys related to {model_name}.")
        saved_model = saved_model[model_name]
        for key in saved_model.keys():
            print(key)
        # del module prefix and backbone prefix because dist train
        state_dict = {}
        for name in saved_model.keys():
            if name.startswith('module.backbone'):
                state_dict[name[16:]] = saved_model[name]
            elif name.startswith('backbone'):
                state_dict[name[9:]] = saved_model[name]
        missing_keys , unexpected_keys = model.load_state_dict(state_dict,strict=False)
    else:
        print(f"The saved model does not contain any keys related to {model_name}.")
        missing_keys , unexpected_keys = model.load_state_dict(saved_model,strict=False)
    print("Missing_keys : ", missing_keys)
    print("Unexpected_keys : ",unexpected_keys)



def load_model_head(model:nn.Module,pretrain_path,model_name):

    saved_model = torch.load(pretrain_path,map_location='cpu') 
    # for key, value in saved_model.items():
    #     print(f"{key}") 

    if any(model_name in key for key in saved_model.keys()):
        print(f"The saved model contains keys related to {model_name}.")
        saved_model = saved_model[model_name]
        state_dict = {}
        for name in saved_model.keys():
            if name.startswith('module.head'):
                state_dict[name[12:]] = saved_model[name]
            elif name.startswith('head'):
                state_dict[name[5:]] = saved_model[name]
    else:
        print("The saved model does not contain any keys related to 'student'.")
    missing_keys , unexpected_keys = model.load_state_dict(state_dict,strict=False)
    print("Missing_keys : ", missing_keys)
    print("Unexpected_keys : ",unexpected_keys)



if __name__ == "__main__":
    load_model_backbone(None,'/public/home/wangshuo/ir_pretrain/dino/dino_vitbase16_pretrain.pth')