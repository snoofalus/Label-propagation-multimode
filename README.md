# Label-propagation-multimode

Utilities and works in progress for a "multimode" use of Label propagation, following the works of Zijan and Ahmet
for a label propagation of a RESNET scaled up to run on concatenated SAR and optical images for possibly better classification than from one image.


TODO:__
-import 1 concatenated image__
-test understanding of prec evaluation, and importance of input variables__
-go through and calculate layers/params for original cifar10 resnet arch__
-create new arch able to run on gpu for concatenated img__
-compare prec on eval with non multimode SAR and optical alone__

IF DONE:__
-create new architecture for multimode learning, 2 conv nets in parallell concatenating features before fully connected layer
