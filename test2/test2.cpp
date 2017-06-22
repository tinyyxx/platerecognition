#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>  
#include <opencv2/video/background_segm.hpp>  
#include<iostream>
#include<string.h>
#include<math.h>
#include<fstream>

//#pragma comment(lib, "opencv_legacy2410d.lib")
using namespace cv;
using namespace std;


Mat stretch (const Mat &image,Mat &result,float x,float y)
{
	float num=0;
	float stretchp[256];
	//清空数组，没初始化是随机的数字
	memset(stretchp,0,sizeof(stretchp));
	//直方图
	MatND hist;
	int channels=0;
	int size=256;
	float hranges[]={0,255};
	const float *ranges[]={hranges};
	calcHist(&image,1,&channels,Mat(),hist,1,&size,ranges);
	//求像素的总个数
	num=image.size().height*image.size().width;
	//求前i个像素级的个数
	for(int i=0;i<256;i++){
		for(int k=0;k<=i;k++)
			stretchp[i]+=hist.at<float>(k);
	}
	//分别求小于0.1和大于0.9概率的值
	int min=0;
	for(;min<256;min++){
		if((stretchp[min]/num)<x)
			break;
	}
	int max=0;
	for(;max<256;max++){
		if((stretchp[max]/num)>y)
		break;
	}
	//创建查找表
	int dim=256;
	Mat lookup(1,&dim ,CV_8U);
	//填充查找表
	for(int i=0;i<256;i++){
		//进行灰度拉伸
		if(i<min)lookup.at<uchar>(i)=0;
		else if(i>max)lookup.at<uchar>(i)=255;
		else lookup.at<uchar>(i)=static_cast<uchar>(255.0*(i-min)/(max-min)+0.5);
	}
	//应用查找表
	LUT(image,lookup,result);
	return result;
}


Mat projectHistogram(const Mat& img ,int t)  //水平或垂直直方图,0为按列统计  
{                                            //1为按行统计  
    int sz = (t)? img.rows: img.cols;  
    Mat mhist = Mat::zeros(1, sz ,CV_32F);  
    for(int j = 0 ;j < sz; j++ )  
    {  
        Mat data = (t)?img.row(j):img.col(j);  
        mhist.at<float>(j) = countNonZero(data);  
    }  
  
    double min,max;  
    minMaxLoc(mhist , &min ,&max);  
  
    if(max > 0)  
        mhist.convertTo(mhist ,-1,1.0f/max , 0);  
  
    return mhist;  
}  
//提取特征函数
void features(const Mat & in , Mat & out)  
{  
    Mat vhist = projectHistogram(in , 1); //水平直方图  
    Mat hhist = projectHistogram(in , 0);  //垂直直方图 

    int numCols = vhist.cols + hhist.cols + in.rows * in.cols;  
    out = Mat::zeros(1, numCols , CV_32F);  
  
    int j = 0;  
    for (int i =0 ;i<vhist.cols ; ++i)  
    {  
        out.at<float>(j) = vhist.at<float>(i);  
        j++;  
    }  
    for (int i=0 ; i < hhist.cols ;++i)  
    {  
        out.at<float>(j) = hhist.at<float>(i);  
    }  
    for(int x =0 ;x<in.rows ;++x)  
    {  
        for (int y =0 ;y < in.cols ;++ y)  
        {  
            out.at<float>(j) = (float)in.at<unsigned char>(x,y);  
            j++;  
        }  
    }  
}  
  

void recog(const Mat &tmpimg1){
	Mat resizeimg;
resize(tmpimg1,resizeimg,Size(480,320));

//转换hsv
Mat hsvimg;
cvtColor(resizeimg,hsvimg,COLOR_BGR2HSV);
//imshow("hsv",hsvimg);

//hsv均衡化
vector<Mat> hsvsplit;
split(hsvimg,hsvsplit);
equalizeHist(hsvsplit[2],hsvsplit[2]); 
Mat mergeimg;
merge(hsvsplit,mergeimg);
//imshow("split",mergeimg);

//检测蓝色区域
Mat blueimg;
inRange(mergeimg,Scalar(100,90,80),Scalar(130,255,255),blueimg);
//imshow("threshimg",blueimg);

//膨胀运算
Mat dilateimg1;
dilate(blueimg,dilateimg1,Mat(2.5,2.5,CV_8U),Point(-1,-1),1);
//开运算
Mat kaiimg;
morphologyEx(dilateimg1,kaiimg,MORPH_OPEN,Mat(3,3,CV_8U),Point(-1,-1),1);
//imshow("开运算",kaiimg);

//灰度化
Mat grayimg;
cvtColor(resizeimg,grayimg,CV_BGR2GRAY);
//imshow("灰度",grayimg);
 
//Mat stretchimg;
//stretch(grayimg,stretchimg,0.15,0.85);
//imshow("拉伸",stretchimg);

//直方图均衡化
Mat equalimg;
equalizeHist(grayimg,equalimg);
//equalizeHist(stretchimg,equalimg);
//imshow("直方图均衡化",equalimg);

//中值滤波
Mat blurimg;
medianBlur(equalimg,blurimg,3); 
//imshow("滤波",blurimg);

//canny
Mat cannyimg;
Canny(blurimg,cannyimg,120,360);
//imshow("canny",cannyimg);

//开闭运算
Mat closeimg;
morphologyEx(cannyimg,closeimg,MORPH_CLOSE,Mat(2,11,CV_8U),Point(-1,-1),1);
Mat openimg;
morphologyEx(closeimg,openimg,MORPH_OPEN,Mat(1,7,CV_8U),Point(-1,-1),1);
//dilate(binaryimg,openimg,Mat(3,3,CV_8U),Point(-1,-1),6); 
//imshow("开闭运算",openimg);

//hsv提取的和边缘检测图像按位与
Mat andimg;
bitwise_and(kaiimg,openimg,andimg);
//imshow("按位与运算",andimg);

//膨胀
Mat dilateimg;
dilate(andimg,dilateimg,Mat(7,7,CV_8U),Point(-1,-1),1);
//imshow("膨胀",dilateimg);

//筛选最大连通域
//查找轮廓
vector<vector<Point>> contours;//元素为点集的向量组
findContours(dilateimg,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
//drawContours(coutoursimg,contours,-1,Scalar(0),2);
//vector<vector<Point>>::const_iterator itcontours= contours.begin();
//寻找最大连通域
vector<Point> contmax;//点集对象
int area,maxarea=0;
//for(;itcontours!=contours.end();++itcontours){
for(int i=0;i<contours.size();i++){
	area=contourArea(contours[i]);//获取当前轮廓的面积
	if(area>maxarea){
		maxarea=area;
		contmax=contours[i];
	}
}
//圈出感兴趣的区域
Rect arect=boundingRect(contmax);
Mat ROIimg,roiimg;
resizeimg.copyTo(ROIimg);
rectangle(ROIimg,arect,Scalar(177,156,242));
//namedWindow("ROI");
imshow("车牌",ROIimg);
roiimg.create(arect.height,arect.width,CV_8U);
resizeimg(arect).copyTo(roiimg);

//精确定位
//精定位提取蓝色区域
Mat HSVimg;
cvtColor(roiimg,HSVimg,COLOR_BGR2HSV);

//namedWindow("hsv");
//imshow("hsv",hsvimg);
//hsv均衡化
vector<Mat> HSVsplit;
split(HSVimg,HSVsplit);
equalizeHist(HSVsplit[2],HSVsplit[2]); 
Mat MERGEimg;
merge(HSVsplit,MERGEimg);
//namedWindow("split");
//imshow("split",mergeimg);
//检测蓝色区域
Mat BLUEimg;
inRange(MERGEimg,Scalar(100,90,90),Scalar(140,255,255),BLUEimg);
//namedWindow("精提取蓝色");
//imshow("精提取蓝色",BLUEimg);
//利用hsv提取蓝色部分，切割车牌

Mat nonzero;
findNonZero(BLUEimg,nonzero);//找出所有非零点集
Rect arect1=boundingRect(nonzero);
//框出蓝色区域
Mat ROIimage;
roiimg.copyTo(ROIimage);
rectangle(ROIimage,arect1,Scalar(177,156,242));
//namedWindow("提取blue");
//imshow("提取blue",ROIimage);

//切割出蓝色车牌
Mat roiimg2;
roiimg2.create(arect1.height,arect1.width,CV_8U);
roiimg(arect1).copyTo(roiimg2);
//namedWindow("切割1");
//imshow("切割1",roiimg2);

//精确定位车牌
Mat ROIgray;
cvtColor(roiimg2,ROIgray,CV_BGR2GRAY);
//Mat threshimg;
//threshold(ROIgray,threshimg,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
//namedWindow("二值化");
//imshow("二值化",threshimg);

//水平倾斜校正
//hough变换，利用houghlineP，概率hough变换
Mat houghimg;
Canny(ROIgray,houghimg,50,200,3);
//imshow("hough canny",houghimg);//canny边缘检测
//hough变换参数设置
int minvote=46;//最小投票数
double minlength=25;// 设置最小线段长度和线段间距容忍度  
double maxgap=4;
double rho=1;
double theta=CV_PI/180;
vector<Vec4i> lines;//返回的坐标值
//hough变换寻找线段
HoughLinesP(houghimg,lines,rho,theta,minvote,minlength,maxgap);

//寻找lines中最大的线段
int n=lines.size();
vector<double> a(n,0);
//double * d=new double[n];//使用new申请动态内存.指针d
int k=0;
//计算长度
	Mat houghimg1;
//roiimg2.copyTo(houghimg1);
if(n!=0){
for(int i=0;i<n;i++){
	Vec4i L=lines[i];
	a[i]=sqrtf((L[0]-L[2])*(L[0]-L[2])+(L[1]-L[3])*(L[1]-L[3]));
	//a[i]=((L[0]-L[2])^2+(L[1]-L[3])^2);
	//cout<<a[i]<<endl;
}

//寻找最大值
double temp=a[0];
for(int i=0;i<n;i++){
	if(a[i]>temp){
		temp=a[i];
		k=i;}
}

Vec4i maxline=lines[k];
//画出此最大线段
Point p1(maxline[0],maxline[1]);
Point p2(maxline[2],maxline[3]);
Mat houghline;;
    roiimg2.copyTo(houghline);
line(houghline,p1,p2,Scalar(177,156,242));
//imshow("hough直线",houghline);

//水平校正
double angle=fastAtan2(maxline[3]-maxline[1],maxline[2]-maxline[0]);//float fastAtan2(float y,float x)
Point2f centerpoint=(ROIgray.cols/2,ROIgray.rows/2);
Mat rotateMat=getRotationMatrix2D(centerpoint,angle,1.0);
//Mat houghimg1;
warpAffine(roiimg2,houghimg1,rotateMat,ROIgray.size(),1,0,0);}

else{roiimg2.copyTo(houghimg1);}

	//delete d;
//imshow("hough校正",houghimg1);


Mat ROIgray1;
cvtColor(houghimg1,ROIgray1,CV_BGR2GRAY);
//imshow("灰度",ROIgray1);
//寻找精确边界
//水平方向  黑白跳变方法  <<基于二值图像的车牌精确定位方法 路小波 张光华>>

//对图像的每一行统计跳变次数,跳变小于14的扫描行不属于车牌区域,
//灰度拉伸
//Mat graystretchimg;
//stretch(ROIgray1,graystretchimg,0.4,0.6);
//imshow("拉升函数",graystretchimg);

//二值化
Mat threshimg;
Mat mean1;  
 Mat stddev1;  
 meanStdDev(ROIgray1,mean1,stddev1 );  
 int T1;
 double t1=sqrt(0.318/0.682);
 T1=mean1.at<double>(0,0)+t1*stddev1.at<double>(0,0);
 threshold(ROIgray1,threshimg,T1,255,CV_THRESH_BINARY);
//threshold(ROIgray1,threshimg,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
//imshow("二值",threshimg);
//统计像素投影
Mat_<uchar> row;
row.create(threshimg.size().height,threshimg.size().width);
int rows=threshimg.size().height;
int cols=threshimg.size().width;
Mat sumrow=Mat::zeros(threshimg.size().height,1,CV_8U);
for(int i=0;i<rows;i++){
	for(int j=0;j<cols;j++){
		threshimg.at<uchar>(i,j)=threshimg.at<uchar>(i,j)/255;//二值化的像素点是0和255！
	}
}

for(int i=0;i<rows;i++){
	for(int j=0;j<cols-1;j++){
		row(i,j)=abs(threshimg.at<uchar>(i,j)-threshimg.at<uchar>(i,j+1));
		sumrow.at<uchar>(i,0)+=row(i,j);
	}
}
//从中间向上下两边搜索
int bottom=rows; 
	int h1=bottom/2,h2=bottom/2;
while (sumrow.at<uchar>(h1,0)>13)
{h1=h1-1;}
while (sumrow.at<uchar>(h2,0)>13)
{h2=h2+1;}
 //切割效果
   int h=h2-h1+1;
   int w=cols;
	Mat roiimg3;
    houghimg1.copyTo(roiimg3);
    //Rect r(top,bottom,left,right);
	rectangle(roiimg3,Rect(0,h1,w,h),Scalar(177,156,242));
   //namedWindow("水平");
   //imshow("水平",roiimg3);
	Mat rowimg;
	houghimg1(Rect(0,h1,w,h)).copyTo(rowimg);
	//namedWindow("水平切割");
 //imshow("水平切割",rowimg);
	//车牌切割
	//基于投影特征值的车牌字符分割算法
  //二值化
   Mat rowgrayimg;
  cvtColor(rowimg,rowgrayimg,CV_BGR2GRAY);
  //基于空间分布的最大类间方差牌照图像二值化算法_张引
Mat rowthreshimg;
Mat mean;  
 Mat stddev;  
 meanStdDev(rowgrayimg,mean,stddev );  
 int T;
 double t=sqrt(0.318/0.682);
 T=mean.at<double>(0,0)+t*stddev.at<double>(0,0);
 threshold(rowgrayimg,rowthreshimg,T,255,CV_THRESH_BINARY);
 // threshold(rowgrayimg,rowthreshimg,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
imshow("切割二值化",rowthreshimg);
  
  //进行垂直投影  草稿

  //二值图投影
  //常规方法
int *shadow=new int[rowthreshimg.cols];
	for(int i=0;i<rowthreshimg.cols;i++){//置零
		shadow[i]=0;
	}
	//归一化
//for(int i=0;i<rowthreshimg.rows;i++){
//for(int j=0;j<rowthreshimg.cols;j++){
	//	rowthreshimg.at<uchar>(i,j)=rowthreshimg.at<uchar>(i,j)/255;//二值化的像素点是0和255！
//	}
//}
	for(int i=0;i<rowthreshimg.cols;i++){
		for(int j=0;j<rowthreshimg.rows;j++){
		shadow[i]+=rowthreshimg.at<uchar>(j,i);
		}
	}
	//for(int i=1;i<rowthreshimg.cols;i++){
		//shadow[i]=shadow[i]/255;
	//}
	 //滤波[0.25,0.5,1,0.5,0.25],进行平滑处理
	//for(int i=2;i<rowthreshimg.cols-2;i++){
	//  shadow[i]=0.25*shadow[i-2]+0.5*shadow[i-1]+1*shadow[i]+0.5*shadow[i+1]+0.25*shadow[i+2];
//  }
	
	Mat cutgrayimg;
	rowthreshimg.copyTo(cutgrayimg);
  //opencv函数reduce投影
 // Mat_<uchar> verMat;  
 // reduce(rowthreshimg,verMat,0,CV_REDUCE_SUM);

	//提取投影的特征值
	int tz=0;
	for(int i=0;i<rowthreshimg.cols;i++){
		tz+=shadow[i];
}
	tz=0.63*(tz/rowthreshimg.cols);

//绘制投影图像
	double maxproj=0;
	for(int i=0;i<rowthreshimg.cols;i++){
		if(shadow[i]>maxproj)
			maxproj=shadow[i];
	}
	//cout<<maxproj<<" ";
	Mat projimg(maxproj,rowthreshimg.cols,CV_8U,Scalar(255));
	for(int i=0;i<rowthreshimg.cols;i++){
		line(projimg,Point(i,maxproj-shadow[i]),Point(i,maxproj-1),Scalar::all(0));
	}
		line(projimg,Point(0,maxproj-tz),Point(rowthreshimg.cols-1,maxproj-tz),Scalar::all(0),15);
	//resize(projimg1,projimg1,Size(rowthreshimg.cols,320));
	//imshow("投影1",projimg1);
	resize(projimg,projimg,Size(480,320));
//imshow("投影",projimg);
	//for(int i=0;i<rowthreshimg.cols;i++){
	//	cout<<shadow[i]<<"    ";}
	
//适用于无粘连的字符切割，效果好
//第一个字符
int start1=0.3*cutgrayimg.cols;//可以找一个更好的方法
	while(shadow[start1]>tz){
		start1=start1+1;
	}
	int end1;
		//寻找上升点
	for(;start1<cutgrayimg.cols;start1++){
		if(shadow[start1]>tz)
		break;
	}
    //寻找下一个等于0的点
	for(end1=start1;end1<cutgrayimg.cols;end1++){
		if(shadow[end1]==0)
		break;
	}
	//寻找起点
	for(;start1>0;start1--){
		if(shadow[start1]==0)
		break;
	}
	int addlength1;
	if(end1-start1+1-0.5*cutgrayimg.rows<0){//需要扩充
		addlength1=0.5*cutgrayimg.rows-(end1-start1+1);}
	else{addlength1=0;}
  Mat zifuimg1;
	rowthreshimg(Rect(start1-addlength1/2,0,end1-start1+1+addlength1,cutgrayimg.rows)).copyTo(zifuimg1);
  //字符归一化
  resize(zifuimg1,zifuimg1,Size(16,32));
  threshold(zifuimg1,zifuimg1,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
 //imshow("字符1",zifuimg1);

  //第二个字符
    int start2=end1;
	int end2;
		//寻找上升点
	for(;start2<cutgrayimg.cols;start2++){
		if(shadow[start2]>tz)
		break;
	}
    //寻找下一个等于0的点
	for(end2=start2;end2<cutgrayimg.cols;end2++){
		if(shadow[end2]==0)
		break;
	}
	//寻找起点
	for(;start2>0;start2--){
		if(shadow[start2]==0)
		break;
	}
  int addlength2;
	if(end2-start2+1-0.5*cutgrayimg.rows<0){//需要扩充
		addlength2=0.5*cutgrayimg.rows-(end2-start2+1);}
	else{addlength2=0;}
  Mat zifuimg2;
	rowthreshimg(Rect(start2-addlength2/2,0,end2-start2+1+addlength2,cutgrayimg.rows)).copyTo(zifuimg2);
  //字符归一化
  resize(zifuimg2,zifuimg2,Size(16,32));
  threshold(zifuimg2,zifuimg2,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
 // imshow("字符2",zifuimg2);
  
  //第三个字符
    int start3=end2;
	int end3;
		//寻找上升点
	for(;start3<cutgrayimg.cols;start3++){
		if(shadow[start3]>tz)
		break;
	}
    //寻找下一个等于0的点
	for(end3=start3;end3<cutgrayimg.cols;end3++){
		if(shadow[end3]==0)
		break;
	}
	//寻找起点
	for(;start3>0;start3--){
		if(shadow[start3]==0)
		break;
	}

	int addlength3;
	if(end3-start3+1-0.5*cutgrayimg.rows<0){//需要扩充
		addlength3=0.5*cutgrayimg.rows-(end3-start3+1);}
	else{addlength3=0;}
  Mat zifuimg3;
	rowthreshimg(Rect(start3-addlength3/2,0,end3-start3+1+addlength3,cutgrayimg.rows)).copyTo(zifuimg3);
  //字符归一化
  resize(zifuimg3,zifuimg3,Size(16,32));
    threshold(zifuimg3,zifuimg3,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
 // imshow("字符3",zifuimg3);

  //第四个字符
    int start4=end3;
	int end4;
		//寻找上升点
	for(;start4<cutgrayimg.cols;start4++){
		if(shadow[start4]>tz)
		break;
	}
    //寻找下一个等于0的点
	for(end4=start4;end4<cutgrayimg.cols;end4++){
		if(shadow[end4]==0)
		break;
	}
	//寻找起点
	for(;start4>0;start4--){
		if(shadow[start4]==0)
		break;
	}
  int addlength4;
	if(end4-start4+1-0.5*cutgrayimg.rows<0){//需要扩充
		addlength4=0.5*cutgrayimg.rows-(end4-start4+1);}
	else{addlength4=0;}
  Mat zifuimg4;
	rowthreshimg(Rect(start4-addlength4/2,0,end4-start4+1+addlength4,cutgrayimg.rows)).copyTo(zifuimg4);
  //字符归一化
  resize(zifuimg4,zifuimg4,Size(16,32));
   threshold(zifuimg4,zifuimg4,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
 // imshow("字符4",zifuimg4);
  
  //第五个字符
    int start5=end4;
	int end5;
		//寻找上升点
	for(;start5<cutgrayimg.cols;start5++){
		if(shadow[start5]>tz)
		break;
	}
    //寻找下一个等于0的点
	for(end5=start5;end5<cutgrayimg.cols;end5++){
		if(shadow[end5]==0)
		break;
	}
	//寻找起点
	for(;start5>0;start5--){
		if(shadow[start5]==0)
		break;
	}
  int addlength5;
	if(end5-start5+1-0.5*cutgrayimg.rows<0){//需要扩充
		addlength5=0.5*cutgrayimg.rows-(end5-start5+1);}
	else{addlength5=0;}
  Mat zifuimg5;
  int zifu5;//判别是否越界
  zifu5=min(end5-start5+1+addlength5,cutgrayimg.cols-start5+addlength5);
  if(start5-addlength5/2+zifu5>cutgrayimg.cols)
  {zifu5=cutgrayimg.cols-start5+addlength5/2;}
	rowthreshimg(Rect(start5-addlength5/2,0,zifu5,cutgrayimg.rows)).copyTo(zifuimg5);
  //字符归一化
  resize(zifuimg5,zifuimg5,Size(16,32));
  threshold(zifuimg5,zifuimg5,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
 // imshow("字符5",zifuimg5);

  //求后几个字符的平均宽度
int meanwidth;
meanwidth=(end1-start1+end2-start2+end2-start2+end4-start4+end5-start5)/5;

  //进行前两个字符的分割
int start6=0.35*cutgrayimg.cols;//可以找一个更好的方法
	while(shadow[start6]>tz){
		start6=start6-1;
	}
	int end6;
		//寻找上升点
	for(;start6>0;start6--){
		if(shadow[start6]>tz)
		break;
	}
    //寻找下一个等于0的点
	for(end6=start6;end6>0;end6--){
		if(shadow[end6]==0)
		break;
	}
	//寻找起点
	for(;start6<cutgrayimg.cols;start6++){
		if(shadow[start6]==0)
		break;
	}
   int addlength6;
	if(start6-end6+1-0.5*cutgrayimg.rows<0){//需要扩充
		addlength6=0.5*cutgrayimg.rows-(start6-end6+1);}
	else{addlength6=0;}
  Mat zifuimg6;
	rowthreshimg(Rect(end6-addlength6/2,0,start6-end6+1+addlength6,cutgrayimg.rows)).copyTo(zifuimg6);
  //字符归一化
  resize(zifuimg6,zifuimg6,Size(16,32));
  threshold(zifuimg6,zifuimg6,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
 // imshow("字符6",zifuimg6);

  //汉字的切割
  int start7=end6;//可以找一个更好的方法
	int end7;
		//寻找上升点
	for(;start7>0;start7--){
		if(shadow[start7]>tz)
		break;
	}
    //寻找下一个等于0的点
	for(end7=start7;end7>0;end7--){
		if(shadow[end7]==0)
		break;
	}
	//寻找起点
	for(;start7<cutgrayimg.cols;start7++){
		if(shadow[start7]==0)
		break;
	}
	//判别汉字的宽度是否满足要求
	int wordwidth;
	wordwidth=start7-end7;
	while(wordwidth<0.8*meanwidth){
		//求下一个为零的波谷
		for(;end7>0;end7--){
		if(shadow[end7]>0)
		break;
	}
		for(end7=end7-1;end7>0;end7--){
		if(shadow[end7]==0)
			break;}
		wordwidth=start7-end7;
		}
   int addlength7;
	if(start7-end7+1-0.5*cutgrayimg.rows<0){//需要扩充
		addlength7=0.5*cutgrayimg.rows-(start7-end7+1);}
	else{addlength7=0;}
  Mat zifuimg7;
	rowthreshimg(Rect(end7-addlength7/2,0,start7-end7+1+addlength7,cutgrayimg.rows)).copyTo(zifuimg7);
  //字符归一化
  resize(zifuimg7,zifuimg7,Size(16,32));
  threshold(zifuimg7,zifuimg7,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
 // imshow("字符7",zifuimg7);/**/

   //汉字识别
	CvANN_MLP bp2;//创建神经网络
	//int x1,x2,x3;
	Mat layerSizes2=(Mat_<int>(1,3)<<560,132,31);//一个3层的神经网络，其中第一层结点数为x1,第二层结点数为x2，第三层结点数为x3
	bp2.create(layerSizes2,CvANN_MLP::SIGMOID_SYM);//构造函数创建
	CvANN_MLP_TrainParams params2;//神经网络训练参数
	//终止条件，迭代次数和误差最小值，一旦有一个达到条件就终止训练。
	params2.term_crit=cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10000, 0.001);
	params2.train_method=CvANN_MLP_TrainParams::BACKPROP; //设置反向传播算法
	params2.bp_dw_scale=0.07;//权值更新率
	params2.bp_moment_scale=0.07;//权值更新冲量
  
//加载训练xml文件
	bp2.load("mlp2.xml");
	//提取目标特征
	Mat zifu7(1 ,560 , CV_32FC1,Scalar(0));
	//Mat zifu60=zifuimg6.reshape(0,1);
	features(zifuimg7, zifu7);  
	Mat output7(1 ,31 , CV_32FC1,Scalar(0)); //1*34矩阵  
	
	bp2.predict(zifu7,output7);
     Point maxLoc0;  
        double maxVal0;  
        minMaxLoc(output7 , 0 ,&maxVal0 , 0 ,&maxLoc0);  
        int zimu0 =  maxLoc0.x; 
		string fhz = "\0";
		switch(zimu0)
     {
      case 0: fhz = "京";break; case 1: fhz = "津";break; case 2: fhz = "冀";break; case 3: fhz = "晋";break;
      case 4: fhz = "蒙";break; case 5: fhz = "辽";break; case 6: fhz = "吉";break; case 7: fhz = "黑";break;
      case 8: fhz = "沪";break; case 9: fhz = "苏";break; case 10: fhz = "浙";break; case 11: fhz = "皖";break;
      case 12: fhz = "闽";break; case 13: fhz = "赣";break; case 14: fhz = "鲁";break; case 15: fhz = "豫";break;
      case 16: fhz = "鄂";break; case 17: fhz = "湘";break; case 18: fhz = "粤";break; case 19: fhz = "桂";break;
      case 20: fhz = "琼";break; case 21: fhz = "渝";break; case 22: fhz = "川";break; case 23: fhz = "贵";break;
      case 24: fhz = "云";break; case 25: fhz = "藏";break; case 26: fhz = "陕";break; case 27: fhz = "甘";break;
      case 28: fhz = "青";break; case 29: fhz = "宁";break; case 30: fhz = "新";break;
    }
		//cout<<zimu0<<endl;
		//char  s2[] = {'京','津','冀','晋','蒙','辽','吉','黑','沪','苏',
//'浙','皖','闽','赣','鲁','豫','鄂','湘','粤','桂','琼','渝','川','贵','云','藏','陕','甘','青','宁','新'}; 
		cout<<fhz;
		/**/

		//字母识别
	CvANN_MLP bp;//创建神经网络
	//int x1,x2,x3;
	Mat layerSizes=(Mat_<int>(1,3)<<560,117,24);//一个3层的神经网络，其中第一层结点数为x1,第二层结点数为x2，第三层结点数为x3
	bp.create(layerSizes,CvANN_MLP::SIGMOID_SYM);//构造函数创建
	CvANN_MLP_TrainParams params;//神经网络训练参数
	//终止条件，迭代次数和误差最小值，一旦有一个达到条件就终止训练。
	params.term_crit=cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30000, 0.001 );
	params.train_method=CvANN_MLP_TrainParams::BACKPROP; //设置反向传播算法
	params.bp_dw_scale=0.07;//权值更新率
	params.bp_moment_scale=0.07;//权值更新冲量



	//神经网络的训练
/*	Mat inputs(1200,560,CV_32FC1,Scalar(0));
	//初始化输入样本
	 ifstream file("muban.txt");  
    int imgindex = 0;
    for (;imgindex<1200;imgindex++)  //表示文件流的结尾
  {  
        char txt_cont[1200];//指针  
        file.getline(txt_cont,1200); //获取字符串
        char imgfile[1200],savefile[1200]; 
        sprintf(imgfile, zimu1/%s", txt_cont);  //连接字符串
		Mat src = imread(imgfile);//注意读取的图片被视为是真彩色图像
		cvtColor(src,src,CV_BGR2GRAY);
		//resize(src,src,Size(32,16));
		threshold(src,src,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
		Mat charfeature(1,560,CV_32FC1,Scalar(0));
		features(src, charfeature);  

		//Mat src1=src.reshape(0,1);
	for(int i=0;i<560;i++)
		{
			inputs.at<float>(imgindex,i)=charfeature.at<float>(0,i);
		}
	}
	/*for(int i=0;i<32;i++){
	   for(int j=0;j<16;j++){
		   cout<<int(src.at<uchar>(i,j))<<" ";
	   }
	   cout<<endl;
   }
	}*/
	
	
//初始化输出样本
	/*Mat outputs(1200,24,CV_32FC1,Scalar(0));
	CvMLData mlData;
    mlData.read_csv("output.csv");//读取csv文件
   outputs= Mat(mlData.get_values(),true);
	
/*for(int i=0;i<50;i++){
	   for(int j=0;j<24;j++){
		   cout<<outputs.at<float>(i,j)<<" ";
	   }
	   cout<<endl;
   }*/
//训练函数的接口,输入矩阵， 预期输出矩阵，训练参数
	//bp.train(inputs,outputs,Mat(),Mat(), params);
	//bp.save("mlp.xml");
//加载训练xml文件
	bp.load("mlp.xml");
	//提取目标特征
	Mat zifu6(1 ,560 , CV_32FC1,Scalar(0));
	//Mat zifu60=zifuimg6.reshape(0,1);
	features(zifuimg6, zifu6);  
	Mat output6(1 ,24 , CV_32FC1,Scalar(0)); //1*34矩阵  
	
	bp.predict(zifu6,output6);
	/*for(int i=0;i<24;i++){
			cout<<output6.at<float>(0,i)<<" ";}
	    cout<<endl;*/
   Point maxLoc;  
        double maxVal;  
        minMaxLoc(output6 , 0 ,&maxVal , 0 ,&maxLoc);  
        int zimu =  maxLoc.x; 
		char  s[] = {'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q',  
        'R','S','T','U','V','W','X','Y','Z'}; 
		cout<<s[zimu]<<" ";
		/**/

		//数字字母识别
	CvANN_MLP bp1;//创建神经网络
	//int x1,x2,x3;
	Mat layerSizes1=(Mat_<int>(1,3)<<560,139,34);//一个3层的神经网络，其中第一层结点数为x1,第二层结点数为x2，第三层结点数为x3
	bp1.create(layerSizes1,CvANN_MLP::SIGMOID_SYM);//构造函数创建
	CvANN_MLP_TrainParams params1;//神经网络训练参数
	//终止条件，迭代次数和误差最小值，一旦有一个达到条件就终止训练。
	params1.term_crit=cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS,30000, 0.001 );
	params1.train_method=CvANN_MLP_TrainParams::BACKPROP; //设置反向传播算法
	params1.bp_dw_scale=0.06;//权值更新率
	params1.bp_moment_scale=0.06;//权值更新冲量

	bp1.load("mlp1.xml");
	//提取目标特征

	//第一个字符
	Mat zifu1(1 ,560 , CV_32FC1,Scalar(0));
	features(zifuimg1, zifu1);  
	Mat output1(1 ,34 , CV_32FC1,Scalar(0)); //1*34矩阵  
	bp1.predict(zifu1,output1);
        Point maxLoc1;  
        double maxVal1;  
        minMaxLoc(output1 , 0 ,&maxVal1,0,&maxLoc1);  
        int zimu1 =  maxLoc1.x; 
		char  s1[] = {'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q',  
        'R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9'}; 
		cout<<s1[zimu1];

		//第二个字符
	Mat zifu2(1 ,560 , CV_32FC1,Scalar(0));
	features(zifuimg2, zifu2);  
	Mat output2(1 ,34 , CV_32FC1,Scalar(0)); //1*34矩阵  
	bp1.predict(zifu2,output2);
        Point maxLoc2;  
        double maxVal2;  
        minMaxLoc(output2 , 0 ,&maxVal2 , 0 ,&maxLoc2);  
        int zimu2 =  maxLoc2.x; 
		cout<<s1[zimu2];

		//第三个字符
	Mat zifu3(1 ,560 , CV_32FC1,Scalar(0));
	features(zifuimg3, zifu3);  
	Mat output3(1 ,34 , CV_32FC1,Scalar(0)); //1*34矩阵  
	bp1.predict(zifu3,output3);
        Point maxLoc3;  
        double maxVal3;  
        minMaxLoc(output3 , 0 ,&maxVal3 , 0 ,&maxLoc3);  
        int zimu3 =  maxLoc3.x; 
		cout<<s1[zimu3];

		//第四个字符
	Mat zifu4(1 ,560 , CV_32FC1,Scalar(0));
	features(zifuimg4, zifu4);  
	Mat output4(1 ,34 , CV_32FC1,Scalar(0)); //1*34矩阵  
	bp1.predict(zifu4,output4);
        Point maxLoc4;  
        double maxVal4;  
        minMaxLoc(output4 , 0 ,&maxVal4 , 0 ,&maxLoc4);  
        int zimu4 =  maxLoc4.x; 
		cout<<s1[zimu4];

		//第五个字符
	Mat zifu55(1 ,560 , CV_32FC1,Scalar(0));
    features(zifuimg5, zifu55);  
	Mat output5(1 ,34 , CV_32FC1,Scalar(0)); //1*34矩阵  
	bp1.predict(zifu55,output5);
        Point maxLoc5;  
        double maxVal5;  
        minMaxLoc(output5 , 0 ,&maxVal5 , 0 ,&maxLoc5);  
        int zimu5 =  maxLoc5.x; 
		cout<<s1[zimu5];
//	 waitKey(0);
    
}  

int main()  
{  
	//string name="video/3test.avi";
	string name1="car99.jpg";
	string name2="video/1test.avi";
	cout<<"识别图片按1；识别视频请按2："<<endl;
	int choose;
	cin>>choose;
  if(choose==1)
	   {  
		Mat picimg;
		picimg=imread(name1);
		recog(picimg);
    } 
  else
	{
	VideoCapture capture(name2);
    if (!capture.isOpened())  
    {  
        cout<<"read video failure"<<std::endl;
		Mat picimg;
		picimg=imread(name1);
		recog(picimg);
        return -1;  
    } 
	
   Mat foreground;  
   Mat background; 
   BackgroundSubtractorMOG2 mog;  
   Mat frame,tmpimg,tmpimg1;
   int backcounter=0;
    long frameNo = 0;  
	for(;;)
	{
		capture>>frame;
		if( frame.empty() )  
            break;  
		
		frame(Rect(Point(300,200),Point(800,530))).copyTo(tmpimg1);
		//frame.copyTo(tmpimg1);
       
        frame(Rect(Point(220,250),Point(720,500))).copyTo(tmpimg);
        rectangle(frame,Point(220,250),Point(720,500),Scalar(255,255,0),1,CV_AA);
		
		//imshow("background",tmpimg); 

		// 运动前景检测，并更新背景  
        mog(tmpimg, foreground, 0.0008);         
         
        // 腐蚀  
      erode(foreground, foreground,Mat());  
        // 膨胀  
       dilate(foreground, foreground,Mat());  
 
        mog.getBackgroundImage(background);   // 返回当前背景图像  
        //cvtColor(foreground,foreground,CV_BGR2GRAY);
  /* threshold(foreground,foreground,0,255,CV_THRESH_OTSU+CV_THRESH_BINARY);
		int counter1 = 0;  
        Mat_<uchar>::iterator it1 = foreground.begin<uchar>();  
        Mat_<uchar>::iterator itend1 = foreground.end<uchar>();    
    for (; it1!=itend1; ++it1)  
    {  
        if((*it1)<100) counter1+=1;//二值化后，像素点是0或者255  
    }    
	if(counter1<1000){
		 for(int x =0 ;x<foreground.rows ;++x)  
    {  
        for (int y =0 ;y < foreground.cols ;++ y)  
        {  
            foreground.at<uchar>(x,y) = 0;  
        }  
    }  
	}*/
    //  imshow("foreground",foreground);
		imshow("video", frame); 
		
     // imshow("background", background); 

	    int counter = 0;  
        Mat_<uchar>::iterator it = foreground.begin<uchar>();  
        Mat_<uchar>::iterator itend = foreground.end<uchar>();    
    for (; it!=itend; ++it)  
    {  
        if((*it)>200) counter+=1;//二值化后，像素点是0或者255  
    }    
  if(backcounter<=80000&&counter>80000)
	{
		recog(tmpimg1);
		backcounter=counter;
		cout<<endl;
		// imshow("foreground1",foreground);
		//break;
		continue;
	}
	backcounter=counter;


	int delay = 50/capture.get(CV_CAP_PROP_FPS);
	waitKey(delay);
 }  
    
	
	capture.release();/**/
	}
	waitKey(0);
    
}  
  