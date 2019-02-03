训练好的模型对象，如何进行传递？

- 场景：1）组件之间；2）notebook训练好的模型传递给组件；3）notebook之间；4）拖拽式训练好的模型传递给notebook(应该比较少见)；5）单机和分布式之间，如单机训练好的模型放到Spark进行大规模批量离线预测，反之亦然；6）机器学习框架之间，如TensorFlow、Pytorch、Keras之间
- 拖拽式训练时Learner组件直接传递给Predictor组件
- 其它场景训练好的模型需持久化到文件。模型存储的文件格式应方便不同场景下模型的可解析、可执行。【文件格式：PMML、MLeap？】



### PMML

PMML是用XML来描述数据挖掘模型的一种通用可交换格式，利用PMML可以将各种工具生成的模型很方便的发布到生产环境。

sklearn 和 R中的模型，都支持导出为PMML格式。

### MLeap

MLeap allows data scientists and engineers to deploy machine learning pipelines from Spark and Scikit-learn to a portable format and execution engine.

Our goals for this project are:

1. Allow Researchers/Data Scientists and Engineers to continue to build data pipelines and train algorithms with Spark and Scikit-Learn
2. Extend Spark/Scikit/TensorFlow by providing ML Pipelines serialization/deserialization to/from a common framework (Bundle.ML)
3. Use MLeap Runtime to execute your pipeline and algorithm without dependenices on Spark or Scikit (numpy, pandas, etc)



### 参考资料

1. [PMML 标准介绍及其在数据挖掘任务中的应用](https://www.ibm.com/developerworks/cn/xml/x-1107xuj/index.html)