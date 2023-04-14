# model folder
- <span style='color:lightgreen'>학습한 모델의 weight와 log를 저장</span>하는 폴더
- <span style='color:pink'>architecture 단위</span>로 관리
    ```python
    CNN
    -> alexnet
        -> alexnet_sgd
            -> model.pth
            -> ~~.log
        -> alexnet_adam
    -> resnet
        -> resnet18_sgd
        -> resnet18_adam
    -> transformer
        -> transformer_sgd
        -> transformer_sgd
    ```