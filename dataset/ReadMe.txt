Each project is stored in a struct of MATLAB.

All of fields and their meanings in a struct are listed as follows.

project.belong        : the dataset that a project belongs to;
project.name          : the name of a project;
project.data          : the data of a project, each row represents a module and each column represents a metric(feature);
project.metric        : the metric set of a project;
project.label         : the labels of modules in a project;
project.numOfInstance : the number of modules;
project.numOfDef      : the number of defective modules;
project.numOfNdef     : the number of non-defective modules;
project.ratioOfDef    : the defective rate.
project.randomidx     : the random sort of indexes of modules, which is used to split the target data randomly.