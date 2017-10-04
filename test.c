int main(){
//unsigned long long x = 0x002000000000000UL;

unsigned long long x = 0x0002000000000000UL;
unsigned int *y= (unsigned int *)&x;
printf("%x %x\n",y[0],y[1]);
}
