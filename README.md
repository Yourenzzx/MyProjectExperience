---
typora-copy-images-to: upload
---

# 项目经历



2019.8-2020.8  TCL工业研究院计算机视觉（AI）算法实习生

- ADC(Auto Defect Classification)项目：缺陷检测->缺陷分类->模板分割->缺陷分割->开单逻辑->系统集成。ADC能定位图片上的Defect做分类，能输出覆盖的pixel数量，能识别defect的几何特性，如：断线，异常相交等，并能依据规则对glass进行判定，如是否需Rework，Repair，开立异常单，等级判定等
  
- 实习学到的：工具、工作方式
  
  - 工具：linux、xshell的隧道（端口映射）、docker 
  - 工作方式：日报周报、沟通。
  
- 实习中最大的困难：竞标华星光电ADC项目。训练集有噪声、在给出的测试集图片在训练集中很少、各类别不平衡的情况下，准确率刷到85%。

  - 【插曲】厂商给出训练集、测试集，其中测试集没有给出类别。我们很单纯地自己将训练集划分成训练与验证集，反复地训练训练集这一批数据。后面研究院经理开会直接说，这其实就是打比赛啊，不要这么老实，要善于利用测试集，把测试集当成验证集用，大家都是这么干的。怎么干的呢？
  - 【tricks】:
    1. 人工判定。研究训练集，摸清训练集不同类别的图片的特征，人工对测试集进行分类，很有把握的，直接人工分出来，加入训练集。最终验收的是一个csv表格，对于人工判定把握的，直接在csv里面填入类别。
    2. 变相向厂线人员咨询。对于训练集中类别有异议的，可以与厂线人员咨询沟通。后来，通常都会把模型判错的、测试集的图片与训练集的图片夹杂，得到测试集图片的真实分类。
    3. 训练集中的噪声剔除。训练好的模型预测训练集，依然判断错误的图片即为无法拟合的图片，剔除。重新训练。一直重复，得到一个干净的数据集，训练出一个baseline
    4. 类别不平衡导致小标注远远多于大标注，沙子类缺陷通常只检出一个点。检测模型置信度与检测框面积加权的方法。
    5. 充分利用图片上的文本信息。记录了缺陷的大小，发现规律，人工分类

- 实习期间遇到比较大的困难：一天内0基础基于tensorflow写出推理代码。

- 覆盖率 = (img_conf > x) / total_img 。其中(img_conf > x)的图片要去除掉判为turn on开单的图片
  
- 心得：
  - 与上级沟通与独立思考很重要，和上级沟通更侧重别人帮你思考。要学会思考别人的想法的可行性（他这么想的目的是为什么，为什么他是这么想的，这种想法具有可行性吗？），在这个基础上，也要学会独立思考，将自己的想法与别人的想法对比、结合。
  - 与同事沟通很重要。
  - 训练数据的重要性
  - 通用能力：合作能力、沟通能力、快速学习能力、解决问题能力
  - 周报、日报的重要性：刚开始，日报周报对我而言只是一个流程性的事情。项目的空闲期，有时候会觉得日报周报写起来很痛苦。因为有这么一个事情的存在，“被迫”养成了一个习惯，工作上的每件小事，都要用markdown井井有条地写下来，这样一来，日报周报就很好写了。但慢慢地，发现了养成这么一个习惯的另一个好处：及时记录下自己的工作的思考、想法——这会极大提升自己工作的效率，同时地也极大提高了同事间沟通的效率。
  - 观察到了老员工在工作态度之间的差异，工作态度好的员工的工作效率确实好很多。
  - 敢于质疑：有时候领导并没有在一线工作，所以具体的一些细节东西不懂。有时候确实会让你做一些“曲线救国”的事，要积极沟通、敢于质疑，争取不做无用功。不要吃没沟通好的亏，臆想领导永远是对的。
  - 研究生期间让自己最自豪的一件事：实验室的大BOSS亲自出面找我谈话让我留在实验室帮他做项目，出“读博去哥伦比亚大学留学的牌”留我。
  - 
  
- 一年实习生涯的成果，2253为本人负责的站点，综合指标TOP1：

  ![](https://github.com/ischansgithub/picture/blob/master/img/各站点抽检指标.jpg)
  
- 2020.6.1-：无须人工标注的图像自动标注与分割（使用AttentionGAN）

  - 遇到的问题：

    - python的不关注类型的习惯到上手C++时遇到的与类型有关的printf问题

      ```cpp
      float f = 1.0;
      int   i = 2;
      printf("f is %d, i is %f", f, i);
      //我觉在这里printf的%d和%f应该不会太影响结果，预期的输出是f is 1, i is 2.0
      //实际上它的输出是f is 2, i is 1
      //这个问题困扰了有一会儿，以为在其他地方变量被修改了导致这种“反过来的错误”
      ```

  - 做过的事情 ：
    - 基于缺陷位置的显著性，以显著性最低的像素为种子，执行二值形态学重建的区域生长，进行缺陷的自动标注与自动分割

    - C语言实现一个图像旋转90度的小算法用于TFT阵列图像的放置方向统一：先求矩阵转置，再沿x轴水平翻转实现旋转90度

    - C语言实现一个图像旋转任意角度的小算法用于TFT阵列图像的放置方向统一：计算旋转中心、输入旋转中心坐标、旋转角度给API计算旋转矩阵、仿射变换

    - 动态文本区域的裁剪算法以适配不同站点的图片：

      - 【伪代码】

        ```python
        def getTxtHeight():
        	1.sobel计算图像导数进行边缘检测	
            2.转为灰度图
            3.计算axis=1的灰度平均值
            4.记下行与行之间灰度为最大差值时的索引idx
            5.该idx即为文字区域的上边界
        ```

      - 【原码】[crop_txt.py](https://github.com/ischansgithub/MyProjectExperience/blob/master/%E9%A1%B9%E7%9B%AE%E7%BB%8F%E5%8E%86%E8%B5%84%E6%96%99/code/crop_txt.py)

      - 【效果】

        ![加载图片失败](https://github.com/ischansgithub/MyProjectExperience/tree/master/项目经历资料/image/txt_crop.png)

    - 使用hough变换来估计TFT阵列中纵贯线相对于垂直直线的偏移角度后矫正模板角度，做角度矫正的目的是为了使模板匹配得更准确。
      - 什么是hough变换【[参考链接](https://blog.csdn.net/sw3300255/article/details/82658596)】：
        - 【应用】hough变换通常用于几何形状特征检测如直线检测。
        - 【图像空间】一条直线的表达式为y=kx+b，通常情况下xy为变量，kb为常数，我们称之为图像空间。
        - 【参数空间，霍夫空间】设定xy为常数，kb为变量，我们称之为参数空间（霍夫空间）。
        - 【两个空间的关系】图像空间的一个点(x0,yo)对应参数空间的一条线，图像空间的一条线对应参数空间的一个点（因为图像空间中一条线的kb已经确定，对应参数空间的点(k,b)）。
        - 【怎么检测直线的？】图像空间n个点对参数空间m条线，如果其中有m条线相交于一点，那么这m条线对应的图像空间的m个点构成一条直线。我们会设定一个阈值，当m大于阈值时，认为这m个点构成一条直线。
        - 【坐标转换】在直角坐标系中，垂直x轴的直线斜率无限大，难以表示。我们通常用极坐标代替直角坐标。用p( theta)=xcos(theta)+ysin(theta)代替y(x)=kx+b，其中p代表原点到直线的距离，theta为直线与x轴的角度。

- 2020.4.22 - 6.1 ： T2-2253站点85吋TFT-LCD站点的开发（12类）：使用改进的yolo_v3进行缺陷检测；使用ResNet进行缺陷分类。

  - 遇到的问题：

    - 【GPU】在docker训练报错：

      ```
      ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
      ```

      根本原因是pytorch会利用共享内存，出现这种问题说明docker的共享内存不够大。直接原因是我在一个docker环境中，训练三个模型，每个模型的dataloader初始化参数的num_workers都设为3

      ```
      dataloader = torch.utils.data.DataLoader(
              dataset,
              batch_size=16,
              shuffle=True,
              num_workers=0,
              pin_memory=True,
              collate_fn=dataset.collate_fn
          )
      ```

      解决方法：将num_workers设为 <= 1 。但是这是一个trade off，会使训练速度变慢

  - 做过的事情：

    - 【检测】：对于新的85吋产品（工艺、玻璃布局的差异，导致85吋图片特征与之前 的d10、d13都不一样）进行迁移训练的试验。方案一是直接使用85吋新产品的小样本数据train from srcatch。方案二是将之前 d10、d13的数据合并

      成5W+，训练一个model后，在此Model上进行85吋产品的finetune。方案三是依据图片特征、特点，将d10+d13所有图片及相应的bounding box旋转90度（使d13+d13数据与85吋产品数据图片特征尽可能相似）后训练model，再基于此Model 微调85吋产品检测模型。
      
    - AttentionGan后处理缺陷自动标注、缺陷分割

- 2020.2.25 -  ：了解TurnOn、开单规则与代码逻辑：C实现算法，用gcc -o lib.so -shared -fPIC  main.c从main.c中生成so库给python调用。makefile的学习。
  
- 2020.1 T2-2253站点正式开发（12类）：使用改进的yolo_v3进行缺陷检测；使用ResNet进行缺陷分类。
  
  - 遇到的问题：
    - 【turnOn 与 分类、检测的trade off关系】优先级最高的任务是不能漏turnon。但是又不能过turnon太多。过turnon的体现在于覆盖率低。覆盖率 = (img_conf > x) / total_img 。其中(img_conf > x)的图片要去除掉判为turn on开单的图片。过turnon太多，分子变小，覆盖率变低。
    
    - 【检测】检出的框太多，影响后续的trunon判断，耗时。解决方法：提高检测模型的阈值。
      
    - 【检测】我须要在检测中解决二分类问题（缺陷类与无缺陷类），但是我发现一段令人疑惑的代码。有这么一个if else：如果一个batch中有bounding box的时候，会计算loss，否则loss会另外处理。当时阅读代码的固有思维是有缺陷的走if分支计算Loss，而没有缺陷的走另一条分支计算loss。但是后面发现一个矛盾的事情，就是在我的数据中，有无缺陷数据都是交融在一起的，也就是说在一个batch中，几乎不可能存在全部数据都为无缺陷的情况！因此也就是说，else分支几乎不会走，那么无缺陷数据不就完全没处理吗！？实际上在收集数据的函数中，有一个小关键点，就是有bound box的会赋予一个索引号，没有bounding box则为空。正是有了这个索引，在if的处理中，没有这个索引的话，有无物品的mask会按默认值，这会使bbox的宽高Loss、中心点loss都会为0，只有预测分数loss会反向传播进行网络参数的更新。因此虽然没有走else这个分支，但是还是会处理无缺陷数据 。
    
    - 小点的缺陷数据太多导致检测模型结果会倾向于框出小框，而大框的分数会比较低。实际上我们也有大片的缺陷，因此解决的办法是final_socre =  score * bbox_width * bbox_length。总结：对检测框进行面积加权+降低置信度阈值。小点缺陷的检出的问题主要的解决思路有三个：提高数据量；调整nms函数；调整Loss。
    
    - 写模型训练日志分析脚本[analyze_txt.py](https://github.com/ischansgithub/MyProjectExperience/blob/master/%E9%A1%B9%E7%9B%AE%E7%BB%8F%E5%8E%86%E8%B5%84%E6%96%99/code/analyze_txt.py)的时候，遇到一个有关Python的语言小问题，检查了好久：
    
      ```python
      a = b = 0 # 本来想简化不同的变量a,b的初始化，但是却遇到了问题
      a += 1
      b += 2
      print(a) # 结果 a = 3  
      ```
    
  - 做过的事情：
    - 小点缺陷数据太多（4000张，其他类平均500张），导致沙状缺陷及划痕缺陷检测出的框会有忽略主要缺陷而框小点缺陷的倾向。解决：将小点缺陷数据减少到550张，从单独训练无缺陷样本的权重文件finetune，训练48个epoch，不改变其他条件下检测分类联调结果84%->86%
    
    - 检测分类联调代码pytorch：写成一个Predict类，初始化函数，获取检测模型函数，检测模型推理，获取分类模型函数，分类模型推理，联合测试函数，画出bbox函数，评估函数（输出混淆矩阵、统计不同分类阈值下的分类准确率、覆盖率、准确率*覆盖率）[pipeline.py](https://github.com/ischansgithub/MyProjectExperience/blob/master/%E9%A1%B9%E7%9B%AE%E7%BB%8F%E5%8E%86%E8%B5%84%E6%96%99/code/pipeline.py)
    
    - 检测分类结果分析代码：用训练好的检测分类模型测试训练集，挑出第二自信度>0.1 、>0.2、及分类错误的图片进行查看分析，旨在能过卡阈值，将一些连训练过都，再一次推理不好判断的数据剔除训练集。
    
    - 缺陷检测模型推理生成xml：节约人工标注的时间[gen_xml_by_yolo_predict.py](https://github.com/ischansgithub/MyProjectExperience/blob/master/%E9%A1%B9%E7%9B%AE%E7%BB%8F%E5%8E%86%E8%B5%84%E6%96%99/code/gen_xml_by_yolo_predict.py)
    
    - 杂乱的事情：根据图片名字筛选是否属于模板图片[isInsidePanel.py]()；将PyTorch的pth转成tensorflow用的pb[To_pb_YOLO.py](https://github.com/ischansgithub/MyProjectExperience/blob/master/%E9%A1%B9%E7%9B%AE%E7%BB%8F%E5%8E%86%E8%B5%84%E6%96%99/code/To_pb_YOLO.py)[To_pb_RESNET.py](https://github.com/ischansgithub/MyProjectExperience/blob/master/%E9%A1%B9%E7%9B%AE%E7%BB%8F%E5%8E%86%E8%B5%84%E6%96%99/code/To_pb_RESNET.py)。
    
    - 新模型上线，准确率提高（74%  -> 89%），但是覆盖率下降（75% -> 65%）,解决覆盖率下降的问题。
    
    - 解决过杀的问题（无须turn on，但却被模型判为turn on）: 找最佳阈值 + 放宽turn on逻辑
    
    - 完成标注文档说明
    
    - 华星光电运行数据的记录表格数据分析脚本：excel表格记录图片名，分类分数，系统判的类别，人工判的类别，是否须要turn on， 缺陷的pixel大小 ，图片名...。该脚本具有通用性，能生成混淆矩阵，有助于分析最佳阈值，提升覆盖率。
    
      ![加载图片失败](https://github.com/ischansgithub/MyProjectExperience/blob/master/%E9%A1%B9%E7%9B%AE%E7%BB%8F%E5%8E%86%E8%B5%84%E6%96%99/image/image-20200326210713784.png)
    
    - 华星光电运行数据的记录表格数据分析脚本:寻找最佳阈值，来优化turn on （TCOTS/TTFBG/TPDPS/TCFBA）逻辑，该脚本能生成表格。
    
    - txt2xml脚本的实现：每一张图片对应一个txt，txt中记录归一化的bbox的中心坐标、长宽。[txt2xml.py](https://github.com/ischansgithub/MyProjectExperience/blob/master/%E9%A1%B9%E7%9B%AE%E7%BB%8F%E5%8E%86%E8%B5%84%E6%96%99/code/txt2xml.py)
    
    - 训练脚本输出的Log文本的解析：Log文本比较杂，主要运用matplotlib.pyplot实现[analyze_txt.py](https://github.com/ischansgithub/MyProjectExperience/blob/master/%E9%A1%B9%E7%9B%AE%E7%BB%8F%E5%8E%86%E8%B5%84%E6%96%99/code/analyze_txt.py)。
    
      ![加载图片失败](https://github.com/ischansgithub/MyProjectExperience/blob/master/%E9%A1%B9%E7%9B%AE%E7%BB%8F%E5%8E%86%E8%B5%84%E6%96%99/image/图片1.png)
  
- 2019.12 T7站点POC 检测分类（7类）：使用改进的yolo_v3进行缺陷检测；使用ResNet进行缺陷分类。
  
  - 遇到的问题：
    - 【检测】IOU的函数没有用对，导致目标检测可视化的结果还不错，但是recall指标却很低。出现这个情况的直接原因是：组长给的代码iou函数没有用对，默认的是输入box是xyxy的标注框，而实际上我们要用的是xywh的yolo标注框。根本原因：刚开始没经验，并且对目标检测yolo的原理理解不足，代码也不熟悉，debug耗费较大的精力，用了一天才debug出问题。
    - 【分类】有两类特征混淆严重。一类特征A是单个小点，另一类特征B是多个小点，整体具有一定的方向性，可连成一条直线。仔细观察数据后，发现是因为标注框的问题。B的标注将多个小点分散成多个标注框，而不是将多个有方向性的小点框成一个大标注，导致分类模型学习不到全局的特征。
  - 做过的事情：
    - 根据目标检测的结果，分析可视化的数据，重新标注。分析可视化的数据，发现框偏小，且距离目标框有一定的偏移性（为什么要根据代码后继的阅读），框偏小的原因是标注的问题，一张图上细碎的小标注框太多。
    - 根据代码crop图片的原则，对图片进行重新标注。不同的crop原则要与具体图片的特征、标注对应。
  
- 2019.11  GLASS/MASK/MURA分类（4类/5类）：使用InceptionResNet-V2进行孔分类。
  
  - 遇到的问题
    
    - 想要进行模型融合，就要比对两个模型的分类分数来评估模型融合的可行性。但是用tfrecord不能得到图片的名称。所以改用源图片进行输入，进行推理。但是图片数据太大，导致tf.concat太慢。直接原因是：我们最开始读入的数据是Numpy，喂入网络的数据也是必须是Numpy，为了使用tf-slim集成的与特定网络（inception-resnet）相关联的图片处理函数（其输入是tensor）,做了件蠢事，将图片的数据类型由numpy->tensor->numpy，推理慢的原因正是在于将tensor-> numpy【tensor.eval()】。根本原因是基础知识不扎实，如果仔细看inception_processing这个图片处理函数的话，就会发现其中的实现完全可以由numpy + opencv替代（比如说，图片数据变成float，以图片为中心进行裁剪, resize_bilinear,归一化处理）。而且当时不太明白，为什么要图片除于255，并乘2 - 1。
    
    - 缺陷目标检测到分类的联合测试接口代码实现：缺陷目标检测的结果为json文件：
    
    ```
      {'1.jpg':[[[x1, y1, x2, y2],confidence1],[[x1, y1, x2, y2],confidence2]...}
    ```
    
     		 检测的结果异常，几乎为同一类。原因是没有对图片进行image = np.divide(np.array(image),  np.float32(255.0))，归一化处理。
    
    - 联合检测带来的问题：我们根据华星光电的缺陷规则人工分类缺陷进行训练，但我们没有考虑到上游项目目标检测中分数较低的缺陷（分数底可能是无缺陷）的注释框尺寸分布、特征分布。导致了推理结果的不理想。
    - 数据分析发现的问题：特征小，输入数据不是299*299时，需要resize成299，一旦检测框大，而缺陷小，很容易造成的后果是把缺陷给压缩没了。尽量使用固定299\*299的框。
    
    
    
  - 做过的事情
    - 根据XML标注文本，在图片上框选出来，存储。同时存储bbox的图片。[draw_and_save_bbox_from_xml.py](https://github.com/ischansgithub/MyProjectExperience/blob/master/%E9%A1%B9%E7%9B%AE%E7%BB%8F%E5%8E%86%E8%B5%84%E6%96%99/code/draw_and_save_bbox_from_xml.py)
    - InceptionResNet-V2的输入固定为299*299。bbox有时候为长方形，resize成299\*299可能 导致特征变形，影响训练结果。所以对bbox重新处理(以长边为基础，长方形变为正方形;并扩大15%的边界;对在边缘的缺陷进行极限处理)
    - 想要进行【模型融合】，就要比对两个模型的分类分数来评估模型融合的可行性。但是用tfrecord不能得到图片的名称。所以改用源图片进行输入，进行推理。
    - 两个模型分别 生成两个csv表格，记录【标签、推理标签、推理的最高分数、图片名】
    - 缺陷目标检测到分类的联合测试接口代码实现：缺陷目标检测的结果为json文件
    - 无缺陷数据的生成：训练集中包含缺陷图片与标注框，首先随机生成defect-free图片的size与左上的坐标，若生成的图片与标注框的IOU大于0，则返回一个标志位，循环执行，直到生成的defect-free图片满足要求。（标注框占比80%的图片不取defect-free图片）
    - 分类阈值评估（数据分析）：根据华星光电测试集结果的部分反馈（这一部分图片知道正确标签），将知道正确标签的图片挑出来，加入相应图片的测试结果（top1/top2 confidence 、及对应的label) ，写成csv，并对所分析图片的top1 confidence进行降序，评估取阈值的可行性。
    - 联合检测分类设定阈值：检测结果分数低于阈值的图片不进行推理，而直接分类为无缺陷类，其他图片送到分类网络；分类检测分数低于阈值的图片直接分为其他类。
    - 训练MASK的分类网络：
      - 心得：基于ImageNet训练的模型进行迁移训练，效果并不显著，由于只会fine-tune最后两层网络，Loss降不下去，一直训练到300000-step，准确率也只有75%   。  
      -  tricks：充分利用图片上的文本信息，可能记录了缺陷的一些信息，有时候有利于我们简化问题或者说是解决问题  。比如T7站点的数据在右下角有缺陷大小的文本，我们可能通过识别这个数字文本来对两种类（两种类缺陷既小又相似，难以区分）进行分类。       
  
- 2019.10  孔分类（4类）：使用InceptionResNet-V2进行孔分类。

  - 遇到的问题

    - GPU训练出的模型，restore保存为pb，预测用本机CPU，报错。加载模型进行预测会报错。怎么解决的呢？耐心地在一大串杂乱的输出中找到，并启发想法，是不是在CPU下面运行的问题呢？

      ```
      attr=use_cudnn_on_gpu:bool,default=true
      ```

      如果要用CPU训练，那么预测的程序要加入以下语句：

      ```python
      os.environ["CUDA_VISIBLE_DEVICES"]="-1"
      ```

    - 从checkpoint恢复出网络和权重后，不知道要往哪个结点塞数据（和pb不一样）。怎么解决的呢？恢复出网络后，用tf.summary.FileWriter保存网络图后，用tensorboard查看网络图，根据数据的输出输出形状找到图片数据的入口。
    
    - 组长运用tf-slim中的预训练模型inception-resnet-v2进行迁移学习，自己生成pb文件，编写恢复Pb/checkpoint文件进行预测的代码，inception-resnet的网络架构限制输入图片只能是299*299，最初直接对自己64\*64的数据集进行imresize()，结果是输出的分类结果恒定只有一类！后来偶然发现了tensorflow/models/research/slim/preprocessing/inception_processing.py这个文件，有专门用于处理图片的一个函数，问题解决。
    
    - 推理测试代码GPU利用率很低，原因是待推理的图片是一张一张遍历喂给网络，而没有进行tf.concat
    
  - 做过的事情：

    - 从Pb/checkpoint文件恢复网络与权重，并进行批量预测。并不断完善脚本，运用PrettyTable，并进行一系列预测数据的统计。
    - 尝试读取tfrecord和源图片喂到从pb恢复的网络中，tfrecord是通过tf.train.batch组合成一个batch喂给网络，FPS=30，而源图片是一张一张遍历喂给网络，FPS=1.2。后来将源图片进行tf.concat，提高了GPU的利用率（23% -> 100%）
    - 小脚本：（预测的脚本会输出文本文件记录预测错误的图片）读取txt文件，对str运用split()进行关键信息的提取，将预测错误的图片单独保存，以便进一步的分析来制定下一个训练策略。
    - 完善推理脚本，推理速度由FPS=0.6 -> 1.8 ->30。

- 摩尔纹数据集的处理：两组成对的图片集，每一组下分辨率各不一致的图片，根据图片的黑白边界找到ROI并裁减。困难的地方在于，如果两组图片集分别裁减，那么最终两两成对的图片分辨率不能完全相同（由于两成对图片的颜色、清晰度略有不同，所以导致一、二个像素的差异）。解决的方法：先裁减一组，将这组的每张图片的裁减参数保存为pkl，再读取pkl文件对另一组图片集进行同等像素的裁减。

- 摩尔纹数据集的处理：比较src和tgt两个文件夹的相对应的图片长宽是否相同。做法是将两个文件夹下面的图片的长宽分别存储于两个列表，两个列表作差，再调用np.nonzero()返回非零元素的索引，从而找到不同长宽的图片。

- 数据集的处理：手动清洗质量较差的图片；用Pillow批量去除在图片上影响训练效果的无关文本；

- 关于GANs的总结PPT制作：分为两类，一类是针对GAN结构上的优化，另一类是针对GAN损失函数的优化。

- CycleGAN模型在集成电路制造中的应用
  
  - 最终目标旨在构建一个集成电路制造中缺陷的AI检测模型，但缺陷检测模型的训练需要大量的已标注的数据。我们已经有了一个自动化标注工具，但是这个自动化标注工具并不可以直接从一张图片直接对缺陷进行标注（不然，这个自动化标注工具不就是我们最终想要的AI模型了吗！），它需要两张图片（一张有缺陷，一张没有缺陷）进行对比，才能进行标注。那么两张成对的图片（一张有缺陷，一张没有缺陷）要哪里来？就是从CycleGAN中来，CycleGAN不需要成对的图片数据，只需要两个域即可，一个域是有缺陷的A，另一个域是没有缺陷的B，CycleGAN就可以从有缺陷的图片生成无缺陷的图片，得到自动化标注工具的成对图片数据。
  
- 摩尔纹数据集的ROI框选
  
  - 现有的摩尔纹的数据共有13万张图片，但每一张图片的分辨率都不尽相同，因此ROI的位置并不都在一个位置，如何框选ROI才能使13万张图片的整体性最好呢？具体问题具体分析，我是通过寻找黑白颜色的跳变点来定位边缘线的位置。
  
- 各种不同形式的GAN的调研

2019.6 - 第六届“星云股份杯”数学建模大赛：视频抄袭检测方法研究

- 定义、提取关键帧：RGB特征差异法
- 推断经亮度处理、增加字幕后的视频是否被盗用：SURF 特征点检测与匹配
- 推断增加噪声后的视频否被盗用：均值哈希算法、汉明距离
- 改进、评估关键帧检测的模型：感知哈希算法与RGB特征差异法融合

2019.4 - 2019.7 基于RGB视频的三维人体重建 

- 项目简介：给定一段视频，第一步分别应用AI预测人体的三维骨架信息及提取人体Mask；第二步，我们结合SMPL人体模型对人体表面进行带有衣物细节及蒙皮的精准重建；第三步，我们将第一步得到的三维骨架信息与第三步重建出的模型进行融合并渲染出三维虚拟人的实时动作。
- 遇到的困难：
  - opendr包，chumpy包的小众。资料少，pip找不到，conda找不到，后面是从源码build，再setup。
- 脚本一：pkl文件的打开脚本
- 脚本二：输入含有掩膜的RGB视频，输出每帧图片
- 脚本三：对每一帧图片进行排序，并整合F帧含有掩膜的二值图，转成hdf5格式（难点：例如存在frame0.png，frame1.png，... ，frame100.png。sorted函数只能对字符串排序，因此得到的排序结果是frame0.png，frame1.png，frame10.png，frame100.png，frame101.png，...  ，sort只能对数值进行排列，所以用sort函数，并传入lamda函数将'.png'屏蔽，得到纯数字字符串后转为Int型,进而排序）



```python
parser = argparse.ArgumentParser()
parser.add_argument('src_folder', type=str)
parser.add_argument('target', type=str)

args = parser.parse_args()

out_file = args.target
mask_dir = args.src_folder

#mask_files = sorted(glob(os.path.join(mask_dir, '*.png')) + glob(os.path.join(mask_dir, '*.jpg')))
##############################   KEY  ##############################################
mask_list=os.listdir(mask_dir)#列出路径下的全部文件的名称
mask_list.sort(key = lambda x: int(x[:-4]))#key = lambda x: int(x[:-4]) 用于屏蔽‘.png’四个字符
mask_files=[]
#print(mask_list)
for i in range(len(mask_list)):
    mask_files.append(mask_dir + mask_list[i])#拼接路径与图片名称
#print(mask_files)
##################################################################################
with h5py.File(out_file, 'w') as f:
    dset = None
for i, silh_file in enumerate(tqdm(mask_files)):
    silh = cv2.imread(silh_file, cv2.IMREAD_GRAYSCALE)

    if dset is None:
        dset = f.create_dataset("masks", (len(mask_files), silh.shape[0], silh.shape[1]), 'b', chunks=True, compression="lzf")

    _, silh = cv2.threshold(silh, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('test',silh)
cv2.waitKey(100)
    dset[i] = silh.astype(np.bool)
```

- 脚本四：格式转化脚本，mat文件转为pkl

2019.2 - 2019.4 基于kinect的无模型动态三维人体重建

- 复现《DynamicFusion——Reconstruction and Tracking of Non-rigid Scenes in Real-Time》
- 非刚体对齐，网格建图
- CUDA

2018.9 - 2019.1 SLAM

- 位姿估计
- 回环检测
- 室内静态场景重建
- 相机标定 

2017.2-2017.6 广州致远电子股份有限公司  |  嵌入式软件工程师

- 项目简介：基于低功耗蓝牙、六轴传感器的无线空中鼠标，在空中感知用户手势，实现与桌面鼠标相同的光标移动、左键右键。

- 遇到的困难：

  - 蓝牙：

    - 用定时器使BLE主从机自动连接是整个“无线飞鼠”源码工程中的难点（官方例程是手动连接，也即通过按键+UART串口从PC发送消息），因为需要把握好定时的时间。为什么这么说呢？在定时器中，我们将扫描设备、连接设备及使能设备这三步顺序执行，也就是说，在定时中断服务函数中，只有执行完了第一个函数也就是扫描设备的函数，才能执行连接设备的函数。但是关键的问题在于ke_schedule()这个消息调度的函数。消息调度函数是当我们用户调用一个服务函数后，QN9021内核会自动调用句柄函数从而进行一系列的操作。所以，当我们调用扫描设备的函数后，内核会自动调用其他函数来实现完整的扫描设备功能。因此，冲突在于，比如当你调用连接设备的函数后，定时器的时间控制不当，ke_schedule()消息自动调度的函数还未执行完毕就执行使能设备的函数，那么，主从设备的连接就会失败。
    - 操作QN9021例程中系统调度黑盒。QN9021例程中，最为核心的函数是ke_schedule(),但是被封装成o文件，看不到实现细节。

  - MPU6050:

    - mpu6050驱动移植中的一个小细节。对MPU6050初始化总共需要进行两次，一次是MPU650本身的初始化，再一次是DMP驱动的初始化。值得注意的是在DMP驱动初始化的函数中，还需要重新对MPU6050本身进行另一些细微的小操作，所以涉及到了两个相似的函数：MPU_Init()与mpu_init()。MPU_Init()用来进行第一次初始化，mpu_init()用来进行DMP初始化之前的再次初始化。假如在DMP初始化之前第二次调用MPU_Init()的话，会出现上位机上模拟出来的姿态与实际上有出入。比如说实际上我们将MPU6050转了半圈，然而上位机上模拟出来已经转了一圈。这说明模拟出来的数据是不正确的。因为之前一直没有发现这个潜在的问题，所以在后来调试的时候，一直在错误数据的基础上一错再错。导致了走了很多弯路。
    - 移植上位机协议的问题：“匿名上位机”是有特定用来模拟MPU6050姿态的功能，所以自然也有自己的一套的协议来实现传感器与上位机的沟通。在官方源码中是以STM32为内核进行编写的，在发送一个字节的函数中，直接操作了STM32的寄存器。而我用的主控芯片是LPC54100，并且搭载了一个AWorks小系统。在Awork中，很多关于寄存器的东西都进行了隐密的封装。所以在挖掘STM32的寄存器与LPC54100的寄存器花费了很大的功夫。最终还是找到了有关的UART发送寄存器AMHW_USART0->txdat和发送状态寄存器AMHW_USART0->stat并加以操作。
    - 最费心的是将MPU6050官方源码模拟I2C时序的程序用AWorks中的接口替换。官方源码是基于STM32进行开发的，并且很多源码是直接牵涉到STM32的底层，更有涉及到STM32的寄存器，所以在移植之前，还需要仔细地阅读官方源码，搞明白官方源码每一行程序究竟做的是什么。最终的移植的确牵涉到了寄存器：在LPC54100(CORTEX-M4)中找到与STM32相对应的寄存器进行源码的移植
    - 鼠标不断地发MPU6050的数据给USB端与蓝牙的连接是会产生干扰，这里也是项目的难点。所以用定时中断，等待一段时间后，也就是等待蓝牙设备连接完毕后，再触发标志位，进而无限循环调用SPI数据发送函数。

  - 其他：

    - 在刚开始的设计方案中，QN9021从设备发送到LPC1765主控芯片的方案是用UART。但是在调试鼠标光标的时候发现光标不停地大范围抖动。然后用虚拟示波器将数据打印出来 ，发现数据波形严重跳动，如下图所示。

      ![img](https://raw.githubusercontent.com/ischansgithub/picture/master/imgwps2.jpg)

      究其原因，可能是因为两个UART中断的相互干扰造成了这种现象。一个UART用来接收QN9021从设备发送过来的数据，另一个UART用来将数据打印到上位机上显示。

      ![img](https://raw.githubusercontent.com/ischansgithub/picture/master/imgwps3.jpg)

      后来临时修改方案，将数据传输方案改为SPI

- 做过什么：

  - 硬件

    - ![](https://raw.githubusercontent.com/ischansgithub/picture/master/img/无线空中鼠标PCB图.png)

      ![](https://raw.githubusercontent.com/ischansgithub/picture/master/img无线空中鼠标实物图.jpg)

    - 硬件方案的设计：鼠标端主控芯片为NXP-LPC54100、传感器为四轴飞行器常用的MPU-6050；USB端主控芯片为NXP-LPC1765；无线低功耗蓝牙芯片为QN9021。

    - PCB电路的设计、打板

  - 软件

    - 数据流传输方案的设计，涉及I2C、SPI、USB、BLE![img](https://raw.githubusercontent.com/ischansgithub/picture/master/imgwps1.jpg)
    - MPU6050六轴传感器驱动移植工作：将基于STM32的模拟I2C源码移植到NXP-LPC54100上
    - C实现OOP：结构体替代class，以定义私有头文件的形式隐藏结构体中的具体实现，以结构体对象作为函数参数的方式实现this指针。















