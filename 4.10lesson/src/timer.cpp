


#include <stdio.h>
#include <stdlib.h>

// 写一个快速排序的函数
void quick_sort(int arr[], int left, int right) {

    int i = left, j = right;
    int tmp = arr[(left + right) / 2];
    while (i <= j) {
        while (arr[i] < tmp)
            i++;
        while (arr[j] > tmp)
            j--;
        if (i <= j) {
            int t = arr[i];
            arr[i] = arr[j];
            arr[j] = t;
            i++;
            j--;
        }
    }
    if (left < j)
        quick_sort(arr, left, j);
    if (i < right)
        quick_sort(arr, i, right);
}