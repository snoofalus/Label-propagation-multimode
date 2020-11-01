# Label-propagation-multimode

Utilities and works in progress for a "multimode" use of Label propagation, following the works of Zijan and Ahmet
for a label propagation of a RESNET scaled up to run on concatenated SAR and optical images for possibly better classification than from one image.


TODO:
-import 1 concatenated image
-test understanding of prec evaluation, and importance of input variables
-go through and calculate layers/params for original cifar10 resnet arch
-create new arch able to run on gpu for concatenated img
-compare prec on eval with non multimode SAR and optical alone

IF DONE:
-create new architecture for multimode learning, 2 conv nets in parallell concatenating features before fully connected layer
