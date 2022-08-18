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
         img_out.at<uchar>(i,j)=(float)0.2126*img.at<Vec3b>(i,j)[2]+(float)0.7152*img.at<Vec3b>(i,j)[1]+(float)0.0722*img.at<Vec3b>(i,j)[0];
      }
   }
   return img_out;
}

//阈值
Mat trsd(Mat img,int high_trsd){
   Mat img_out=Mat::zeros(img.rows,img.cols,CV_8UC1);
   for(int i=0;i<img_out.rows;i++){
      for(int j=0;j<img_out.cols;j++){
         if(img.at<uchar>(i,j)>high_trsd){
            img_out.at<uchar>(i,j)=255;
         }
      }
   }
   return img_out;
}


//大津二值化
int Otsu(Mat img){
   int trsd=0;
   int N[256]={0};
   float N_sum=0;
   float Pf=0,Pb=0,Mf=0,Mb=0;
   float M=0,max_var=0;
   float var[256]={0};

   for(int i=0;i<img.rows;i++){
      for(int j=0;j<img.cols;j++){
         N[img.at<uchar>(i,j)]++;
         N_sum++; 
      }
   }
   
   for(trsd=1;trsd<255;trsd++){
      for(int a=0;a<=255;a++){
         if(a<trsd){
            Pf+=(float)N[a]/N_sum;
         }
         else{
            Pb+=(float)N[a]/N_sum;
         }
      }
      //避免分母出现0
      if(Pb!=0&&Pf!=0){
      for(int b=0;b<=255;b++){
            if(b<=trsd){
               Mf+=(float)N[b]*(float)b/Pf;
            }
            else{
               Mb+=(float)N[b]*(float)b/Pb;
            }     
         } 
      
      M=Pf*Mf+Pb*Mb;
      var[trsd]=Pf*(Mf-M)*(Mf-M)+Pb*(Mb-M)*(Mb-M);
      }
      else var[trsd]=0;
      //更新最大值
      max_var=fmax(var[trsd],max_var);
      //初始化
      M=0;
      Pf=0;
      Pb=0;
      Mf=0;
      Mb=0;
      
   }
   for(int c=1;c<255;c++){
      if(max_var==var[c]){
         //遍历返回最大值
         return c;
      }
   }
   return 0;
}
//膨胀
Mat Dlt(Mat img){
   Mat img_out=Mat::zeros(img.rows,img.cols,CV_8UC1);
   for(int i=0;i<img.rows;i++){
      for(int j=0;j<img.cols;j++){
         if(img.at<uchar>(i,j)==0){
            if(i!=0)if(img.at<uchar>(i-1,j)==255)img_out.at<uchar>(i,j)=255;
            if(j!=0)if(img.at<uchar>(i,j-1)==255)img_out.at<uchar>(i,j)=255;
            if(i!=img.rows-1)if(img.at<uchar>(i+1,j)==255)img_out.at<uchar>(i,j)=255;
            if(j!=img.cols-1)if(img.at<uchar>(i,j+1)==255)img_out.at<uchar>(i,j)=255;
         }
      }
   }
   return img_out;
}

int main(void)
{
   
   Mat srcImage=imread("imori.jpg");
   Mat dstImage=switch3To1(srcImage);
   Mat trsd_img=trsd(dstImage,Otsu(dstImage));
   Mat dlt_img=Dlt(Dlt(trsd_img));
   //大津二值化
   imwrite("answer_4.jpg",trsd_img);
   //两次膨胀
   imwrite("answer_47.jpg",dlt_img);
   
   
   
   return 0;
}
