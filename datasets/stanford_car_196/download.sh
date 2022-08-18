#################################
#   Download `car_devkit.tgz`   # 
#################################
wget "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"
tar -xvzf "./car_devkit.tgz"

##################################################
#   Download `cars_test_annos_withlabels.tgz`    # 
##################################################
!wget "https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat"
!mv "/content/dataset/cars_test_annos_withlabels.mat" "/content/dataset/devkit/"

#################################
#   Download `cars_train.tgz`   # 
#################################
%cd "/content/dataset"
wget "https://ai.stanford.edu/~jkrause/car196/cars_train.tgz"
tar -xvzf "/content/dataset/cars_train.tgz"

#################################
#   Download `cars_test.tgz`    # 
#################################
!wget "https://ai.stanford.edu/~jkrause/car196/cars_test.tgz"
!tar -xvzf "/content/dataset/cars_test.tgz"