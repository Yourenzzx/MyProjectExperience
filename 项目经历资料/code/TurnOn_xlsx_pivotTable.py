# -*- coding:UTF-8 -*-
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import glob
import pdb

RootDir = '/home-ex/tclhk/chenww/t2/online_excel/'  # 基礎路徑
xlsxName = 'xlsx/2250-D13-20200219.xlsx'  # 放置xlsx的目錄
# xlsxName = 'xlsx/2250-D13-20200217.xlsx'
my_sheet = 'Sheet1'  # xlsx中的表名稱！！！ 一定要是Sheet1
confidenceThreshold = 60
USE_DIF_CONFIDENCE = 1  # 不同类别采用不同的门限
if USE_DIF_CONFIDENCE == 1:
    ExcludeCode = 'TSFAS'  # 误检测框，没有缺陷
    ThrdStepValue = 5  # 查找门限的步长
    ThrdMin = 30  # 查找门限下限
    ThrdMax = 75  # 查找门限上限

Display_len = 5  # top5
MaxDataLen = 12  # Top12
rowNum = Display_len + 2  # +1: total display +2: histogram
xlsxFolderPath = os.path.join(RootDir, xlsxName)
xlsxPathList = glob.glob(xlsxFolderPath)
SaveFolder = os.path.join(RootDir, 'xlsxAnalysis')
# if os.path.exists(SaveFolder):
#     shutil.rmtree(SaveFolder)
# os.makedirs(SaveFolder)

for xlsxPath in xlsxPathList:
    print("********** {} ***********".format(xlsxPath))
    splitxlsxName = os.path.splitext(xlsxPath)[0]
    splitsuffix = os.path.splitext(xlsxPath)[1]
    if splitsuffix == '.csv':
        xlsxPath = splitxlsxName + '.csv'
        df = pd.read_csv(xlsxPath, encoding='utf-8')
    elif splitsuffix == '.xlsx':
        df = pd.read_excel(xlsxPath, sheet_name=my_sheet)
    else:
        print('Wrong file test in {}'.format(xlsxPath))

    tmp = splitxlsxName.split('/')
    tmp = tmp[-2:]
    titleName = ('_').join(tmp)
    titlePath = os.path.join(SaveFolder, titleName)
    SaveImgName = titlePath + '.png'
    font_size = 6  # 字体大小
    fig_size = (12, 10)  # 图表大小
    plt.rcParams['figure.figsize'] = fig_size
    # 更新字体大小
    plt.rcParams['font.size'] = font_size

    TurnOnImgSaveFolder = os.path.join(RootDir, 'SaveFolder')
    if os.path.exists(TurnOnImgSaveFolder):
        shutil.rmtree(TurnOnImgSaveFolder)
    else:
        os.makedirs(TurnOnImgSaveFolder)
    imgSubFolder = ['TurnTruth', 'TurnFalse']
    TurnOnImgSaveFolder0 = os.path.join(TurnOnImgSaveFolder, imgSubFolder[0])
    TurnOnImgSaveFolder1 = os.path.join(TurnOnImgSaveFolder, imgSubFolder[1])

    os.makedirs(TurnOnImgSaveFolder0)  # 實際需要turnon的圖片
    os.makedirs(TurnOnImgSaveFolder1)  # 實際不需要turnon的圖片

    pt = pd.pivot_table(df, index=['RTIR_DEFECT_NO'], columns=['DEFECT_CODE'], values=['GLASS_ID'],
                        aggfunc={'GLASS_ID': 'count'}, margins=True)

    GTList = df.loc[:, 'RTIR_DEFECT_NO']  # 按标签选择
    GTStatistic = pd.value_counts(GTList)
    GTClassList = GTStatistic.index
    GTClassedNum = len(GTClassList)

    # Step1: code 更新   ～～～～～～～～～～～～～～～～
    for idx, item in enumerate(GTClassList):
        if item == 'TPDPC' or item == 'TPDPL':
            items = ['TPDPC', 'TPDPL']
            df = df.replace('TPDPL', 'TPDPC')

        elif item == 'TTFBO':
            items = ['TCOTS', 'TTFBO']
            df = df.replace('TTFBO', 'TCOTS')
    GTList = df.loc[:, 'RTIR_DEFECT_NO']
    GTStatistic = pd.value_counts(GTList)
    GTClassList = GTStatistic.index
    GTClassedNum = len(GTClassList)

    # =================================Step2: 畫出GT分布統計圖：=============================================
    data = GTStatistic[:MaxDataLen]
    labels = GTClassList[:MaxDataLen]
    index = np.arange(len(data))
    WholeNumber = sum(GTStatistic)
    # 设置柱形图宽度
    bar_width = 0.2

    # 一个figure中分成rowNum行、2列，创建第1个子图 ： 横轴（类别）+ 竖轴（数目），并标出不同的GT_CLS的占比、数目
    plt.subplot(rowNum, 2, 1)
    axs = plt.gca()
    rects = plt.bar(index + bar_width, data, tick_label=labels, color='g', width=bar_width, align="center")
    plt.ylabel("Class Img Number")
    plt.title(titleName + ' WholeNum=' + str(WholeNumber))

    plt.rcParams['figure.figsize'] = (12, 20)  # 图表大小
    # 更新字体大小
    plt.rcParams['font.size'] = font_size

    # 每一个柱状上的标识
    def add_labels(rects, WholeNumber):
        DLen = 0
        MaxV = rects[0].get_height()
        upShift = MaxV / 10
        for rect in rects:
            DLen += 1
            height = rect.get_height()
            precentH = np.round(height / WholeNumber * 100, 1)
            plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
            plt.text(rect.get_x() + rect.get_width() / 2, height + upShift, precentH, ha='center', va='bottom')

            rect.set_edgecolor('white')


    add_labels(rects, WholeNumber)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # =================================Step3: 給出 TurnOn 佔比======================================================
    # 一个figure中分成rowNum行、2列，创建第3个子图 : 每个类别在不同的分数区间的分布
    plt.subplot(rowNum, 2, 3)

    TurnOnNum = sum(df.loc[:, 'CONFIDENCE'] == 1)  # 1 means turn on
    message = 'The TurnOn percentage is ' + str(round(TurnOnNum / WholeNumber * 100, 1)) + '%' + '\n' + 'Top ' + str(
        Display_len) + ' Hist'
    plt.title(message)
    ColorBox = ['red', 'yellow', 'green', 'lime', 'magenta']
    ArrayConf = np.array(df.loc[:, 'CONFIDENCE'])
    labels = GTClassList[:Display_len]

    bins = 10  # x轴区间分布
    ArrayHistList = []
    labels_acc = []
    axs = plt.gca()  # Get Current Axes 获取当前子图
    # Step4: 判斷成TTP3S 而實際GT也是TTP3S的正確率
    for idx, item in enumerate(labels):
        DTLabel = df.loc[df['DEFECT_CODE'] == item]  # 从DataFrame中筛选出 “模型分类结果” == “我们想要的label” 的所有信息
        Pos = DTLabel.loc[:, 'RTIR_DEFECT_NO'] == item  # 进而得到是否 "模型分类结果" == “实际分类结果” 的One-hot向量。
        DTConf = DTLabel.loc[:, 'CONFIDENCE'][Pos]  # 进而得到每个 "模型分类结果" == “实际分类结果”条目的分类分数
        ArrayHistList.append(DTConf)
        # 目标label的准确率
        labels_acc.append(str(labels[idx]) + ':' + str(round(sum(Pos == True) / len(Pos) * 100, 1)))


    n, bins, patches = axs.hist(ArrayHistList, bins, normed=1, color=ColorBox, lw=0, label=labels_acc)
    axs.legend(prop={'size': 6})
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # 一个figure中分成rowNum行、2列，创建第4个子图 :
    plt.subplot(rowNum, 2, 4)
    # 计算Top5正确率，对比：<无门限情况下，以及有门限情况下>
    SumNo = 0
    SumWithThreshold = 0
    RemainNum = 0  # 超过门限的值
    TopNumNo = 0
    TopNumWT = 0
    for idx, item in enumerate(labels):
        # 筛选出 gt_cls = 目标Label 的条目
        TTP3SGT = df.loc[df['RTIR_DEFECT_NO'] == item]
        #  gt_cls = 目标Label 的条目的累加和
        TopNumNo += len(TTP3SGT)
        # 筛选出 pred_cls = 目标Label 的条目
        CorrectList = TTP3SGT.loc[:, 'DEFECT_CODE'] == item
        #  符合conf > thres 的 pred_cls  = 目标Label 的条目的累加和
        CorrectNum = sum(TTP3SGT.loc[:, 'CONFIDENCE'][CorrectList] >= confidenceThreshold)
        SumNo += CorrectNum

        # 所有 conf > thres 的条目的数目累加和
        ThresholdList = TTP3SGT.loc[:, 'CONFIDENCE'] >= confidenceThreshold
        RemainNum += sum(ThresholdList)


        TopNumWT += sum(ThresholdList)
        CorrectList = TTP3SGT.loc[:, 'DEFECT_CODE'] == item
        CorrectList = CorrectList[ThresholdList]
        CorrectNum = sum(CorrectList)  # 目标label == pred_cls 的数目
        SumWithThreshold += CorrectNum

    CoverRate = str(round(RemainNum / TopNumNo * 100, 1))  # 覆盖率
    AveNo = str(round(SumNo / TopNumNo * 100, 1))  # 大于阈值的准确率
    AveTreshold = str(round(SumWithThreshold / TopNumWT * 100, 1))
    message = 'WTO_Acc:' + AveNo + '%  {CR:' + CoverRate + '% WT_Acc:' + AveTreshold + '%}' '\n' + 'Top ' + str(
        Display_len) + ' Hist'
    plt.title(message)
    bins = 10
    ArrayHistList = []
    labels_acc = []
    axs = plt.gca()
    # Step5: GT是TTP3S 而實際Predict也是TTP3S的正確率
    for idx, item in enumerate(labels):
        DTLabel = df.loc[df['RTIR_DEFECT_NO'] == item]
        Pos = DTLabel.loc[:, 'DEFECT_CODE'] == item
        DTConf = DTLabel.loc[:, 'CONFIDENCE'][Pos]
        ArrayHistList.append(DTConf)
        labels_acc.append(str(labels[idx]) + ' Acc:' + str(round(sum(Pos == True) / len(Pos) * 100, 1)))

    n, bins, patches = axs.hist(ArrayHistList, bins, normed=1, color=ColorBox, lw=0, label=labels_acc)
    axs.legend(prop={'size': 6})
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # Step 补充： 查找分类最佳门限
    # Step5: GT是TTP3S 而實際Predict也是TTP3S的正確率
    plt.subplot(rowNum, 2, 2)
    axs = plt.gca()
    message = ''
    RecoredConfv = []
    WholePosNum = 0
    WholeANum = 0
    WholeNum = 0
    temp_labels = ['TCOTS', 'TCPIA', 'TCPOA', 'TCSAD', 'TGGS0', 'TPDPS', 'TSDFS', 'TSILR', 'TTFBG', 'TTP3G', 'TTSPG', 'TSFAS']
    # for idx, item in enumerate(labels):
    for idx, item in enumerate(temp_labels):
        GTLabel = df.loc[df['RTIR_DEFECT_NO'] == item]  # 真实A
        GTPost = GTLabel.loc[:, 'DEFECT_CODE'] == item  # 真A/判A  --TP
        GTConf = GTLabel.loc[:, 'CONFIDENCE'][GTPost]
        GTConfWhole = GTLabel.loc[:, 'CONFIDENCE']

        DTLabel = df.loc[df['DEFECT_CODE'] == item]  # 判A
        DTPost = ~(DTLabel.loc[:, 'RTIR_DEFECT_NO'] == item)  # 判A/真B  --FP
        DTConf = DTLabel.loc[:, 'CONFIDENCE'][DTPost]  # 判A/真B 的confidence
        F1Score = 0

        if item in ExcludeCode:
            RecordTmp = item + ' ' + str(0) + ' '
            RecoredConfv.append(RecordTmp)
            continue

        # 遍历每一个类别，遍历各个thres，查找计算分类最佳门限 -> 相当于找最大F1 Score
        print('========={}==========='.format(item))
        for ConfidenceThV in range(ThrdMin, ThrdMax, ThrdStepValue):
            TPNum = sum(GTConf >= ConfidenceThV)  # 判成A，真是A的数量  -- TP
            TrueA = sum(GTConfWhole >= ConfidenceThV)  # 目标label中所有大于thres的数目
            FPNum = sum(DTConf >= ConfidenceThV)  # 判成A，但是B的数量 -- FP
            ConvRate = TPNum / len(GTLabel) * 100
            if (TPNum + FPNum) == 0:
                PrecRate = 0
            else:
                PrecRate = TPNum / TrueA * 100  # (目标lable中， >thres条件下) 类别预测正确的概率

            # 计算最大F1-score
            F1Tmp = ConvRate * PrecRate / (ConvRate + PrecRate)
            print("{} {}".format(ConfidenceThV, ConvRate))
            if F1Tmp >= F1Score:
                F1Score = F1Tmp
                RecordTmp = item + ' ' + str(ConfidenceThV) + ' ' + str(round(ConvRate, 1)) + ' ' + str(
                    round(PrecRate, 1))

                BestPosNum = TPNum
                BestTrueA = TrueA
        # assert False
        WholePosNum += BestPosNum
        WholeANum += BestTrueA
        WholeNum += len(GTLabel)
        RecoredConfv.append(RecordTmp)

    # 类别的"信息"列表RecoredConfv 合成一个 message
    for RcMsg in RecoredConfv:
        message += RcMsg + '\n'
    message += ' Wl_Cov: ' + str(round(WholeANum / WholeNum * 100, 1)) + ' Wl_Acc: ' + str(
        round(WholePosNum / WholeNum * 100, 1))
    # print(message)
    plt.title(message)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.set_xticks([])
    axs.set_yticks([])

    # Step6: 畫出top5的分類統計圖
    idx = plt.gcf().number

    idx = 4
    data = GTStatistic[:Display_len]
    labels = GTClassList[:Display_len]
    for item in labels:
        idx += 1
        plt.subplot(rowNum, 2, idx)
        axs = plt.gca()
        TTP3SPredict = df.loc[df['RTIR_DEFECT_NO'] == item]
        PredictList = pd.value_counts(TTP3SPredict.loc[:, 'DEFECT_CODE'])
        Subindex = np.arange(len(PredictList))
        LenMax = min(len(PredictList.values), 10)
        rects1 = plt.bar(Subindex[:LenMax] + bar_width, list(PredictList.values[:LenMax]),
                         tick_label=list(PredictList.index[:LenMax]), color='g', width=bar_width, align="center")
        plt.ylabel("Class Img Number")
        plt.title(item)
        WholeNumber = sum(PredictList)
        add_labels(rects1, WholeNumber)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        # with confidencethreshold
        idx += 1
        plt.subplot(rowNum, 2, idx)
        axs = plt.gca()
        ConfiIdx = TTP3SPredict.loc[:, 'CONFIDENCE'] >= confidenceThreshold
        RemainNum = sum(ConfiIdx)
        CoverRate = str(round(RemainNum / len(TTP3SPredict) * 100, 1))  # 覆盖率
        TurnOn = TTP3SPredict.loc[:, 'CONFIDENCE'] == 1
        TurnOnNum = np.sum(TurnOn == True)
        PredictList = pd.value_counts(TTP3SPredict.loc[:, 'DEFECT_CODE'][ConfiIdx])
        Subindex = np.arange(len(PredictList))
        LenMax = min(len(PredictList.values), 10)
        rects1 = plt.bar(Subindex[:LenMax] + bar_width, list(PredictList.values[:LenMax]),
                         tick_label=list(PredictList.index[:LenMax]), color='g', width=bar_width, align="center")
        plt.ylabel("Class Img Number")
        plt.title(item + '_cofTh=' + str(confidenceThreshold) + ': TONum = ' + str(TurnOnNum) + ' CR:' + CoverRate)
        WholeNumber = sum(PredictList)
        add_labels(rects1, WholeNumber)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        # plt.show()
    # 收工
    plt.savefig(SaveImgName)
    plt.close()











