# MVControl-threestudio

[**Paper**](https://arxiv.org/abs/xxxx.xxxxx) | [**Project Page**](https://lizhiqi49.github.io/MVControl/)

Official implementation of **Controllable Text-to-3D Generation via Surface-Aligned Gaussian Splatting**

[Zhiqi Li](https://github.com/lizhiqi49), [Yiming Chen](https://github.com/codejoker-c), [Lingzhe Zhao](https://github.com/LingzheZhao), [Peidong Liu](https://ethliup.github.io/)

**The code will be released later.**

Abstract: *While text-to-3D and image-to-3D generation tasks have received considerable attention, one important but under-explored field between them is controllable text-to-3D generation, which we mainly focus on in this work. To address this task, 1) we introduce Multi-view ControlNet (MVControl), a novel neural network architecture designed to enhance existing pre-trained multi-view diffusion models by integrating additional input conditions, such as edge, depth, normal, and scribble maps. Our innovation lies in the introduction of a conditioning module that controls the base diffusion model using both local and global embeddings, which are computed from the input condition images and camera poses. Once trained, MVControl is able to offer 3D diffusion guidance for optimization-based 3D generation. And, 2) we propose an efficient multi-stage 3D generation pipeline that leverages the benefits of recent large reconstruction models and score distillation algorithm. Building upon our MVControl architecture, we employ a unique hybrid diffusion guidance method to direct the optimization process. In pursuit of efficiency, we adopt 3D Gaussians as our representation instead of the commonly used implicit representations. We also pioneer the use of SuGaR, a hybrid representation that binds Gaussians to mesh triangle faces. This approach alleviates the issue of poor geometry in 3D Gaussians and enables the direct sculpting of fine-grained geometry on the mesh. Extensive experiments demonstrate that our method achieves robust generalization and enables the controllable generation of high-quality 3D content.*

<p align="center">
    <img src="assets/teaser.png">
</p>


## Method Overview
<p align="center">
    <img src="assets/diagram.png">
</p>



## BibTeX

```bibtex
@misc{li2023mvcontrol,
      title={MVControl: Adding Conditional Control to Multi-view Diffusion for Controllable Text-to-3D Generation}, 
      author={Zhiqi Li and Yiming Chen and Lingzhe Zhao and Peidong Liu},
      year={2023},
      eprint={2311.14494},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

