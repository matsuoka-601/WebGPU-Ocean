#include <stdio.h>
#include <math.h>

int main() {
    for (int i = 0; i < 100; i++) {
        int original = 99; // 試したい整数
        float f = original; // 整数を float に代入
        float ceiled = ceil(f); // ceil を適用

        // printf("Original: %d\n", original);
        // printf("Float: %f\n", f);
        // printf("Ceil: %f\n", ceiled);

        if (original == (int)ceiled) {
            // printf("一致します。\n");
        } else {
            printf("一致しません。\n");
        }
    }


    return 0;
}
