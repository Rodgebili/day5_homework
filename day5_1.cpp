#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

//三通道转化为单通道（灰度图）
Mat switch3To1(Mat img){
   Mat img_out=Mat::zeros(img.rows,img.cols,CV_8UC1);
   for(int i=0;i<img.rows;i++){
      for(int j=0;j<img.cols;j++){
         img_out.at<uchar>(i,j)=0.2126*img.at<Vec3b>(i,j)[2]+0.7152*img.at<Vec3b>(i,j)[1]+0.0722*img.at<Vec3b>(i,j)[0];
      }
   }
   return img_out;
}
//基于已知条件的高斯滤波
Mat Gaosi(Mat img){
   float K[5][5];
   float cvlt=0;//卷积和
   
   Mat img_out=Mat::zeros(img.rows,img.cols,CV_8UC1);
   for(int a=0;a<5;a++){
      for(int b=0;b<5;b++){
         K[a][b] = 1/(2 * M_PI * 1.4 * 1.4)*exp(-((a-2)*(a-2)+(b-2)*(b-2))/(2*1.4*1.4));
         cvlt+=K[a][b];
      }
   }
   for(int aa=0;aa<5;aa++){
      for(int bb=0;bb<5;bb++){
         K[aa][bb]=K[aa][bb]/cvlt;
      }
   }
   
   for(int i=0;i<img.rows;i++){
      for(int j=0;j<img.cols;j++){
         //单通道        
            //卷积
            cvlt=0;
            for(int x=-2;x<=2;x++){
               for(int y=-2;y<=2;y++){
                  if((i+x<=-1)||(j+y<=-1)||(i+x>=img.rows)||(j+y>=img.cols)){cvlt+=0;}
                  else cvlt+=img.at<uchar>(i+x,j+y)*K[x+2][y+2]; 
               }
            }

            img_out.at<uchar>(i,j)=cvlt;
            cvlt=0;
 
      }
   }

   return img_out;
}
//flag=1纵向，=0横向
Mat Sobel_smk(Mat img,bool flag){
   Mat img_out=Mat::zeros(img.rows,img.cols,CV_8UC1);
   int K[3][3];
   //横向卷积核
   int trans[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};
   //纵向卷积核
   int port[3][3]={{1,0,-1},{2,0,-2},{1,0,-1}};
   //根据标志判断横向或纵向
   if(flag){
      for(int a=0;a<3;a++){
         for(int b=0;b<3;b++){
            K[a][b]=trans[a][b];
         }
      }
   }
   else{
      for(int a=0;a<3;a++){
         for(int b=0;b<3;b++){
            K[a][b]=port[a][b];
         }
      }
   }

   int cvlt=0;
   for(int i=0;i<img.rows;i++){
      for(int j=0;j<img.cols;j++){
         //单通道        
            //卷积
            cvlt=0;
            for(int x=-1;x<=1;x++){
               for(int y=-1;y<=1;y++){
                  if((i+x<=-1)||(j+y<=-1)||(i+x>=img.rows)||(j+y>=img.cols)){cvlt+=0;}
                  else cvlt+=img.at<uchar>(i+x,j+y)*K[x+1][y+1]; 
               }
            }
            //将取值范围固定在0-255间
            cvlt=fmax(cvlt,0);
            cvlt=fmin(cvlt,255);
            img_out.at<uchar>(i,j)=cvlt;
            cvlt=0;
      }
   }
   return img_out;
}

//计算边缘梯度
Mat edge_get(Mat img_tra,Mat img_por){
   Mat img_out=Mat::zeros(img_tra.rows,img_tra.cols,CV_8UC1);
   int edge_this;
   for(int i=0;i<img_out.rows;i++){
      for(int j=0;j<img_out.cols;j++){
         //平方和开方
         edge_this=sqrt(img_tra.at<uchar>(i,j)*img_tra.at<uchar>(i,j)+img_por.at<uchar>(i,j)*img_por.at<uchar>(i,j));
         //固定范围
         edge_this=fmax(edge_this,0);
         edge_this=fmin(edge_this,255);
         img_out.at<uchar>(i,j)=edge_this;
      }
   }
   return img_out;
}

//计算角度
Mat angle_get(Mat img_tra,Mat img_por){
   Mat angle=Mat::zeros(img_tra.rows,img_tra.cols,CV_8UC1);
   //fx_fy表示fy/fx
   double fx_fy=1.0;
   double tan_arc=0.0;
   for(int i=0;i<angle.rows;i++){
      for(int j=0;j<angle.cols;j++){
         fx_fy=(double)img_por.at<uchar>(i,j)/(double)img_tra.at<uchar>(i,j);
         tan_arc=atan(fx_fy);//计算arctan
         //确定角度
         if(tan_arc<=-0.4142){angle.at<uchar>(i,j)=135;}
         else if(tan_arc<=0.4142){angle.at<uchar>(i,j)=0;}
         else if(tan_arc<=2.4142){angle.at<uchar>(i,j)=45;}
         else angle.at<uchar>(i,j)=90;
      }
   }
   return angle;
}


int main(void)
{
   
   Mat srcImage=switch3To1(imread("imori.jpg"));
   
   imwrite("edge.jpg",edge_get(Sobel_smk(Gaosi(srcImage),1),Sobel_smk(Gaosi(srcImage),0)));
   imwrite("angle.jpg",angle_get(Sobel_smk(Gaosi(srcImage),1),Sobel_smk(Gaosi(srcImage),0)));
   waitKey(0);
   return 0;
}

