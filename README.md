# VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks (ICLR 2025)

This repo contains the code and data for [VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks](https://arxiv.org/abs/2410.05160). In this paper, we aimed at building a unified multimodal embedding model for any tasks.  

<img width="1432" alt="abs" src="figures/teaser.png">
<a target="_blank" href="https://arxiv.org/abs/2410.05160">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-black?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://github.com/TIGER-AI-Lab/VLM2Vec">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<a target="_blank" href="https://tiger-ai-lab.github.io/VLM2Vec/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/collections/TIGER-Lab/vlm2vec-6705f418271d085836e0cdd5">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/collections/TIGER-Lab/vlm2vec-6705f418271d085836e0cdd5">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/spaces/TIGER-Lab/MMEB">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Leaderboard-red?style=flat"></a>
<a target="_blank" href="https://x.com/WenhuChen/status/1844577017930694984">
<img style="height:22pt" src="https://img.shields.io/badge/-Tweet-blue?style=flat&logo=twitter"></a>
<br>

---

## 🔥News
- [2025-06] 🔥 We’re excited to announce that VLM2Vec_V2 is coming soon! The existing VLM2Vec_V1 codebase will be moved to the ```v1``` branch.
- [2025-02] 🔥 Two new VLM2Vec models are released, based on Qwen2VL 2B and 7B, achieving 60.1 and 65.8 (new SOTA!) on the MMEB benchmark. Check them out ([2B]([url](https://huggingface.co/TIGER-Lab/VLM2Vec-Qwen2VL-2B)), [7B]([url](https://huggingface.co/TIGER-Lab/VLM2Vec-Qwen2VL-7B)))!
- [2025-01] 🎉 **VLM2Vec is accepted to ICLR 2025.**
- [2024-10] The technical report, code, data, and model for VLM2Vec are all available online.

<details>
  <summary>📜 View Older Updates</summary>

- [2025-01] We have updated our [training data](https://huggingface.co/datasets/TIGER-Lab/MMEB-train). Each subset now contains two splits: ```original``` and ```diverse_instruction```. The ```original``` split is provided to support the reproduction of our paper results. The ```diverse_instruction``` split includes paraphrased instructions for each task, designed to enhance instruction diversity and improve the model's robustness to unseen instructions and tasks. Moving forward, our future releases will primarily use the ```diverse_instruction``` split.
- [2024-12] We have released the [MMEB leaderboard](https://huggingface.co/spaces/TIGER-Lab/MMEB). Feel free to contact us if you want to include your model.
- [2024-12] We have released a new variant of VLM2Vec built on the LLaVa-Next backbone, which is currently our best-performing version: https://huggingface.co/TIGER-Lab/VLM2Vec-LLaVa-Next.
- [2024-10] VLM2Vec has been integrated into [vLLM](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_vision_language_embedding.py).

</details>

## Model
Our model is based on converting an existing well-trained VLM into an embedding model. The basic idea is to take the last token in the end of the sequence as the representation of the multimodal inputs. Our VLM2Vec framework is compatible with any SOTA open-source VLMs. By leveraging diverse training data—encompassing a variety of modality combinations, tasks, and instructions—it generates a robust universal multimodal embedding model.

<img width="1432" alt="abs" src="figures/train_vlm.png">

 - [VLM2Vec-Phi3.5V](https://huggingface.co/TIGER-Lab/VLM2Vec-Full)
 - [VLM2Vec-LLaVa-Next](https://huggingface.co/TIGER-Lab/VLM2Vec-LLaVa-Next)
 - [VLM2Vec-Qwen2VL-2B](https://huggingface.co/TIGER-Lab/VLM2Vec-Qwen2VL-2B)
 - [VLM2Vec-Qwen2VL-7B](https://huggingface.co/TIGER-Lab/VLM2Vec-Qwen2VL-7B) (**Current best version VLM2Vec**)


## Data
Our model is being trained on MMEB-train (20 tasks) and evaluated on MMEB-eval (20 IND tasks and 16 OOD tasks).
 - [Train data](https://huggingface.co/datasets/TIGER-Lab/MMEB-train)
 - [Eval data](https://huggingface.co/datasets/TIGER-Lab/MMEB-eval)
<img alt="abs" src="figures/MMEB_Overview.jpg">


## Experimental Results
Our model can outperform the existing baselines by a huge margin.
<img alt="abs" src="figures/vlm2vec_results.png">

## Quick Start
We have provided several samples, including demonstration and evaluation code, located in the `examples/` directory.


## Training

Download the image file zip from huggingface
```
git lfs install
git clone https://huggingface.co/datasets/TIGER-Lab/MMEB-train
cd MMEB-train
python unzip_file.py
cd ../
```

For GPUs with small memory, use GradCache to reduce memory usage, i.e. setting small values to `--gc_q_chunk_size` and `--gc_p_chunk_size`.

Below is a demo training script. You’ll need to set up the full list of ```subset_name``` and use a large batch size.
```bash
torchrun --nproc_per_node=2 --master_port=22447 --max_restarts=0 train.py \
 --model_name Qwen/Qwen2-VL-2B-Instruct --bf16 --pooling last \
 --dataset_name TIGER-Lab/MMEB-train \
 --split_name original \
 --subset_name ImageNet_1K N24News HatefulMemes InfographicsVQA ChartQA Visual7W VisDial CIRR NIGHTS WebQA MSCOCO \
 --num_sample_per_subset 50000 \
 --image_dir MMEB-train \
 --image_resolution high --max_len 4096 \
 --output_dir $OUTPUT_DIR --logging_steps 1 \
 --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 2000 \
 --warmup_steps 200 --save_steps 1000 --normalize True \
 --temperature 0.02 --per_device_train_batch_size 8 \
 --grad_cache True --gc_q_chunk_size 2 --gc_p_chunk_size 2 \
 --save_safetensors False
```

## Inference & Evaluation

Download the image file zip from huggingface
```bash
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip
unzip images.zip -d eval_images/
```


```bash
python eval.py --model_name Qwen/Qwen2-VL-7B-Instruct --checkpoint_path TIGER-Lab/VLM2Vec-Qwen2VL-7B \
  --model_backbone qwen2_vl \
  --encode_output_path outputs/ \
  --pooling last --normalize True \
  --lora True \
  --dataset_name TIGER-Lab/MMEB-eval \
  --subset_name N24News CIFAR-100 HatefulMemes VOC2007 SUN397 ImageNet-A ImageNet-R ObjectNet Country211 \
  --dataset_split test --per_device_eval_batch_size 16 \
  --image_dir eval_images/
```

## Acknowledgement
- We have adapted code from [Tevatron](https://github.com/texttron/tevatron), a flexible and efficient toolkit that supports training and inference for neural retrieval models.


## Citation
```
@article{jiang2024vlm2vec,
  title={VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks},
  author={Jiang, Ziyan and Meng, Rui and Yang, Xinyi and Yavuz, Semih and Zhou, Yingbo and Chen, Wenhu},
  journal={arXiv preprint arXiv:2410.05160},
  year={2024}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TIGER-AI-Lab/VLM2Vec&type=Date)](https://star-history.com/#TIGER-AI-Lab/VLM2Vec&Date)
