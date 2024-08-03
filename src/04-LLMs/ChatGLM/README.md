We utilized the [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) model with `ptuning` for fine-tuning and evaluation. 

We made specific changes to the following scripts to tailor them to our task:

- `ptuning/train.sh`
- `ptuning/evaluate.sh`

Other than the above modifications, all other components remain unchanged. 

For detailed instructions and further information, please refer to the original [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) repository.