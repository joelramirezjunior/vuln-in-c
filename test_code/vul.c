#include <stdio.h>
#include <string.h>

int main(){
    char buffer[10];
    char input[20];
    printf("Enter your name: ");  
    gets(input);
    strcpy(buffer, input);
    printf("Hello, %s!\n", buffer);
    return 0;
}