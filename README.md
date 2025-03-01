
# **Vision Transformer with Local Attention 🚀**

🔍 **A self-implementation of Vision Transformer (ViT) with Local Attention Mechanism.**  

## **🌟 Features**
✅ Patch Embedding with Learnable Tokens  
✅ Multi-Head **Global** and **Local** Self-Attention  
✅ Mix **Global** and **Local** Self-Attention Mechanism within the same model  
✅ Add **CLS** Token to the local attention window of each patch
✅ Transformer Encoder with Configurable Layers  
✅ Customizable Vision Transformer for Various Image Sizes  
✅ Designed for GPU Acceleration 🚀  

## **📝 Note**
❗ **Important:** Using Local Attention Mechanism in Vision Transformer makes the computations much faster and efficient but does not reduce the complexity (number of parameters) of the model.

## **📂 Repository Structure**
```
📁 ViT
 ├── ViT.py               # Main implementation of ViT with Local Attention
 ├── README.md            # You are here! 📜
 ├── examples/            # Example scripts (Coming Soon)  
 ├── experiments/         # Training and testing experiments (Coming Soon)  
```

## **⚡ Quick Start**
```python
from ViT import VisionTransformer

model = VisionTransformer(
    img_size=32, patch_size=4, in_channels=3, num_classes=10,
    embed_dim=128, num_heads=4, num_layers=6, mlp_dim=256,
    dropout=0.1, window_size=[3, 3, 5, 5, None, None], add_cls_token=True
)
print(model)
```

## **📜 License & Usage Notice**
❗ **Important:** This repository is currently **not licensed**.  
If you use any part of this code, please **provide credit** and notify me via [GitHub Issues](https://github.com/komil-parmar/ViT_with_Local_Attention/issues) or email.  
💬 Friendly Note: If you're just experimenting, learning, or using this code for research, no need to add any formalities while contacting me!  
Feel free to explore and have fun with it. 
A formal open-source license will be added later.  

## **🚀 Future Work**
- 📌 Add training and fine-tuning scripts  
- 📌 Implement efficient local attention techniques  
- 📌 Implement variable sized cls token (n patches)
- 📌 Release benchmark results  

## **💬 Contributing**
💡 Suggestions and improvements are welcome! Feel free to open an issue or contribute via pull requests.  

## **👨‍🎓 About Me**
Hi! I'm a college student, self-learning deep learning and AI.
I truly appreciate any kind of feedback, reviews, or suggestions, as they help me grow! 😊

📌 You can learn more about me on my GitHub: [GitHub Profile](https://github.com/Komil-parmar/Komil-parmar)
