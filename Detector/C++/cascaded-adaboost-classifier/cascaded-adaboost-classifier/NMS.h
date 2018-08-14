#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

static void sort(int n, const vector<float> x, vector<int> indices)
{
    // 排序函数，排序后进行交换的是indices中的数据
    // n：排序总数// x：带排序数// indices：初始为0~n-1数目

    int i, j;
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
        {
            if (x[indices[j]] > x[indices[i]])
            {
                //float x_tmp = x[i];
                int index_tmp = indices[i];
                //x[i] = x[j];
                indices[i] = indices[j];
                //x[j] = x_tmp;
                indices[j] = index_tmp;
            }
        }
}


void nonMaximumSuppression(int numBoxes, vector<CvPoint>& points, vector<CvPoint>& oppositePoints, float overlapThreshold, vector<int>& is_suppressed, vector<int>& Nums)
{
    // 实现检测出的矩形窗口的非极大值抑制nms
    // numBoxes：窗口数目// points：窗口左上角坐标点// oppositePoints：窗口右下角坐标点// score：窗口得分
    // overlapThreshold：重叠阈值控制// numBoxesOut：输出窗口数目// pointsOut：输出窗口左上角坐标点
    // oppositePoints：输出窗口右下角坐标点// scoreOut：输出窗口得分
    int i, j;
    vector<float> box_area(numBoxes);             // 定义窗口面积变量并分配空间
    vector<int> indices(numBoxes);                    // 定义窗口索引并分配空间
    // 初始化indices、is_supperssed、box_area信息
    for (i = 0; i < numBoxes; i++)
    {
        indices[i] = i;
        is_suppressed[i] = 0;
        box_area[i] = (float)( (oppositePoints[i].x - points[i].x + 1) *(oppositePoints[i].y - points[i].y + 1));
    }
    // 对输入窗口按照分数比值进行排序，排序后的编号放在indices中
    sort(numBoxes, box_area, indices);
    for (i = 0; i < numBoxes; i++)                // 循环所有窗口
    {
        if (!is_suppressed[indices[i]])           // 判断窗口是否被抑制
        {
            for (j = i + 1; j < numBoxes; j++)    // 循环当前窗口之后的窗口
            {
                if (!is_suppressed[indices[j]])   // 判断窗口是否被抑制
                {
                    int x1max = max(points[indices[i]].x, points[indices[j]].x);                     // 求两个窗口左上角x坐标最大值
                    int x2min = min(oppositePoints[indices[i]].x, oppositePoints[indices[j]].x);     // 求两个窗口右下角x坐标最小值
                    int y1max = max(points[indices[i]].y, points[indices[j]].y);                     // 求两个窗口左上角y坐标最大值
                    int y2min = min(oppositePoints[indices[i]].y, oppositePoints[indices[j]].y);     // 求两个窗口右下角y坐标最小值
                    int overlapWidth = x2min - x1max + 1;            // 计算两矩形重叠的宽度
                    int overlapHeight = y2min - y1max + 1;           // 计算两矩形重叠的高度
                    if (overlapWidth > 0 && overlapHeight > 0)
                    {
                        float overlapPart = (overlapWidth * overlapHeight) / box_area[indices[j]];    // 计算重叠的比率
                        if (overlapPart > overlapThreshold)          // 判断重叠比率是否超过重叠阈值
                        {
                            is_suppressed[indices[j]] = 1;           // 将窗口j标记为抑制
                            Nums[indices[i]] = Nums[indices[i]]+1;
                        }
                    }
                }
            }
        }
    }
}
