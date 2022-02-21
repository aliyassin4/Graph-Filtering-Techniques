/*******************************
Global Statistical Significance (GloSS) filter
Program written by Filippo Radicchi
*********************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>



void count_nodes(char *filename, int *MM);
void read_netw(char *filename, int **edges, double *ww);
double directed_stats(int **edges, double *ww, int E, double *kin, double *kout, double *sin, double *sout, int N);
double undirected_stats(int **edges, double *ww, int E, double *kin, double *kout, double *sin, double *sout, int N);
double calculate_distr_weights(int E, double *ww, double *xx, double *prob, int Q, double kmax, double *sin, double *sout, int N);
void calculate_power(int N, int k, double *prob, double *prod, double **table);
void calculate_inverse_fourier(double **table, int N, int k, double dx, int *deg_tr);
double pvalue(double s1, int k1, double s2, int k2, double w, double **table, double *xx, int N);
void calculate_pvalue(int **edges, double *ww, int E, double *kin, double *kout, double *sin, double *sout, int num_points, double **table, double *xx, double *pp, double dx);
void print_table(char *filename, double **table, double *xx, int k, int N);
void print_rank(char *filedata, int **edges, double *ww, double *pp, int E);
int optimalQ (int E, double *ww, int N, double *sin, double *sout);




void print_msg(char *text)
{
  printf("%s",text);
  fflush(stdout);
}




int main (int argc, char **argv)
{


 

  if(argc!=3) {
    printf("\n#########\n\n");
    printf("The correct usage is:\n./GloSS_continuous edge_list d\n\n");
    printf("edge_list should be a three columns file listing in start node, end node and weight of the connection\n");
    printf("d can be set equal to 1 for the analysis of undirected networks ");
    printf("or 2 for directed ones\n");
    printf("\n#########\n\n\n");
    exit(0);
  }

  int DIR=atoi(argv[2]);
  if(DIR!=1 && DIR!=2) {
    printf("\n#########\n\n");
    printf("The second argument must be 1 or 2, depending if you want analyze an undirected or directed graph\n");
    printf("\n#########\n\n\n");
    exit(0);
  }



  printf("\n#########\n\n");
  printf("Global Statistical Significance (GloSS) filter\n");
  if(DIR==1) printf("\nReading the undirected network %s\n\n",argv[1]);
  if(DIR==2) printf("\nReading the directed network %s\n\n",argv[1]);
  



  int MM[2];
  int N, E;
  count_nodes(argv[1], MM);
  N=MM[0];
  E=MM[1];
  printf("## Tot nodes %d\n",N);
  printf("## Tot edges %d\n",E);
  int **edges, i, j;
  double *ww, *pp;
  ww=(double *)malloc((E+1)*sizeof(double));
  pp=(double *)malloc((E+1)*sizeof(double));
  edges=(int **)malloc((E+1)*sizeof(int *));
  for(i=1;i<=E;i++) edges[i]=(int *)malloc(2*sizeof(int));
  read_netw(argv[1], edges, ww);




  double *kin, *kout, *sin, *sout, kmax;
  kin=(double *)malloc((N+1)*sizeof(double));
  kout=(double *)malloc((N+1)*sizeof(double));
  sin=(double *)malloc((N+1)*sizeof(double));
  sout=(double *)malloc((N+1)*sizeof(double));
  


  if(DIR==1) kmax=undirected_stats(edges, ww, E, kin, kout, sin, sout, N);
  if(DIR==2) kmax=undirected_stats(edges, ww, E, kin, kout, sin, sout, N);
  printf("## kmax %g\n",kmax);



  int *deg_tr;
  deg_tr=(int *)malloc((int)kmax*sizeof(int));
  for(i=0;i<kmax;i++) deg_tr[i]=1;
  for(i=1;i<=N;i++)
    {
      deg_tr[(int)kin[i]-1]++;
      deg_tr[(int)kout[i]-1]++;
    }



 


  int Q=20;
  int num_points=pow(2.0,Q);
  double *prob, *xx, dx;
  prob=(double *)malloc(num_points*sizeof(double));
  xx=(double *)malloc(num_points*sizeof(double));


  printf("## parameter Q %d\n",Q);


  dx=calculate_distr_weights(E, ww, xx, prob, Q, kmax, sin, sout, N);
  double Err=0.0;
  for(i=1;i<=E;i++) if(ww[i]<dx) Err+=1.0;
  printf("## Resolution %g\n",dx);
  printf("## Fraction of edges not distinguished at this resolution %g\n",Err/(double)E);
  if(Err>0) {
    printf("\n\n!!!!\n");
    printf("If you want to be more accurate you should increment the resolution parameter Q\nNotice that high values of Q require high memory usage\n");
    printf("!!!!\n\n\n");
  }

 



  



  //char text[100];
  //sprintf(text,"Number of Points %d\n dx %g\n",num_points,dx);
  //print_msg(text);




  print_msg("## Calculating Fourier Transform : ");
  gsl_fft_real_radix2_transform(prob, 1, num_points);
  print_msg("done\n");


  double *prod, **table;
  prod=(double *)malloc(num_points*sizeof(double));
  table=(double **)malloc((int)kmax*sizeof(double *));
  for(i=0;i<(int)kmax;i++) table[i]=(double *)malloc(num_points*sizeof(double));


  print_msg("## Calculating Powers : ");
  calculate_power(num_points, (int)kmax, prob, prod, table);
  print_msg("done\n");

  print_msg("## Calculating Inverse Fourier Transform : ");
  calculate_inverse_fourier(table, num_points, (int)kmax, dx, deg_tr);
  print_msg("done\n");


  
  print_msg("## Calculating p-values : ");
  calculate_pvalue(edges, ww, E, kin, kout, sin, sout, num_points, table, xx, pp, dx);
  print_msg("done\n");

  
  print_msg("## Writing file : ");
  print_rank(argv[1], edges, ww, pp, E);
  print_msg("done\n\n");



  for(i=1;i<=E;i++) free(edges[i]);
  free(edges);
  free(ww);
  free(kin);
  free(kout);
  free(sin);
  free(sout);
  free(prob);
  free(xx);
  for(i=0;i<(int)kmax;i++) free(table[i]);
  free(table);
  free(prod);
  free(pp);
  free(deg_tr);

  return 0;
}















double pvalue(double s1, int k1, double s2, int k2, double w, double **table, double *xx, int N)
{
  if(k1==1 || k2==1) return -1;
  int i, ss1=-1, ss2=-1, ww=-1, ss1w=-1, ss2w=-1;
  for(i=0;i<N;i++)
    {
      if(ss1<0.0 && xx[i]>=s1) ss1=i;
      if(ss2<0.0 && xx[i]>=s2) ss2=i;
      if(ss1w<0.0 && xx[i]>=s1-w) ss1w=i;
      if(ss2w<0.0 && xx[i]>=s2-w) ss2w=i;
      if(ww<0.0 && xx[i]>=w) ww=i;
    }
 
 
  int q1, q2;
  double wi;
  double cc=0.0, bb=0.0;
  for(i=0;i<N;i++)
    {
      wi=xx[i];
      q1=ss1-i;
      q2=ss2-i;
      //printf("i %d   wi %g   w %g  q1 %d   s1-wi %g   q2 %d  s2-wi %g : %g %g %g\n",i,wi,w,q1,s1-wi,q2,s2-wi,table[0][i],table[k1-2][q1],table[k2-2][q2]);
      //printf("%d %g %g  %d %g  %d %g\n",i,wi,w,q1,s1-wi,q2,s2-wi);
      if(q1>1 && q2>1)
    {
      if(i>=ww) cc+=table[0][i]*table[k1-2][q1]*table[k2-2][q2];
      bb+=table[0][i]*table[k1-2][q1]*table[k2-2][q2];
    }

      
      else{goto endloop;}
    }
  endloop:
  
  //printf("\n");


  if(bb>0.0) return cc/bb;
  return -10.0;
}





void calculate_pvalue(int **edges, double *ww, int E, double *kin, double *kout, double *sin, double *sout, int num_points, double **table, double *xx, double *pp, double dx)
{
  double pval, rpval, w;
  double s1, s2;
  int i, k1, k2, n1, n2;
  int C=0;
  for(i=1;i<=E;i++)
    {
      n1=edges[i][0];
      n2=edges[i][1];
      k1=kout[n1];
      k2=kin[n2];
      s1=sout[n1];
      s2=sin[n2];
      w=ww[i];
      //pval=pvalue(s1, k1, s2, k2, w, table, xx, num_points);
      //rpval=pvalue(s1, k1, s2, k2, 0.0, table, xx, num_points);
      //pp[i]=pval/rpval;
      if(k1>1 && k2>1) pp[i]=pvalue(s1, k1, s2, k2, w, table, xx, num_points);
      if(pp[i]<=0.01) C++;
      if(k1==1 || k2==1) pp[i]=1.0;
    }


  //printf("C %d   %g\n",C,(double)C/(double)E);
}









void calculate_power(int N, int k, double *prob, double *prod, double **table)
{
  int i, r, v;
  double re1, im1, re2, im2, a, b, c, d;
  for(i=0;i<N;i++) prod[i]=prob[i];
  for(i=0;i<N;i++) table[0][i]=prod[i];
  for(v=1;v<k;v++)
    {

      for(i=0;i<=N/2;i++)
    {
      r=N-i;

      //printf("%d %d\n",i,r);

      re1=prob[i];
      im1=0.0;
      if(r<N && i!=N/2) im1=prob[r];
     
      re2=prod[i];
          im2=0.0;
          if(r<N && i!=N/2) im2=prod[r];


      a=re1*re2;
      b=im1*im2;
      c=re1*im2;
      d=re2*im1;


      prod[i]=a-b;
      if(r<N && r!=N/2) prod[r]=c+d;

    }
     
      for(i=0;i<N;i++) table[v][i]=prod[i];
    }
 
}




void calculate_inverse_fourier(double **table, int N, int k, double dx, int *deg_tr)
{
  int i, j;
  double norm;
  for(i=0;i<k;i++)
    {
      if(deg_tr[i]>0){
    gsl_fft_halfcomplex_radix2_inverse (table[i], 1, N);
    norm=0.0;
    for(j=0;j<N;j++) if(table[i][j]<0.0) table[i][j]=0.0;
    for(j=0;j<N;j++) norm+=table[i][j];
    for(j=0;j<N;j++) table[i][j]=table[i][j]/norm/dx;
    norm=0.0;
    for(j=0;j<N;j++) norm+=table[i][j]*dx;
    //printf("i %d  norm %g\n",i,norm);
      }
    }
}
                   






void count_nodes(char *filename, int *MM)
{
  int q, i, j, N=0, E=0;
  double k;
  FILE *f;
  f=fopen(filename,"r");
  while(!feof(f))
    {
      q=fscanf(f,"%d %d %lf",&i,&j,&k);
      if(q<=0) goto exitfile;
      if(i>N) N=i;
      if(j>N) N=j;
      E++;
    }
 exitfile:
  fclose(f);
  MM[0]=N;
  MM[1]=E;
}



///////////////////////////////////////////////////////

void read_netw(char *filename, int **edges, double *ww)
{
  int q, i, j, E=0;
  double k;
  FILE *f;
  f=fopen(filename,"r");
  while(!feof(f))
    {
      q=fscanf(f,"%d %d %lf",&i,&j,&k);
      if(q<=0) goto exitfile;
     

      E++;
      edges[E][0]=i;
      edges[E][1]=j;
      ww[E]=k;

    

    }
 exitfile:
  fclose(f);
 
}




double undirected_stats(int **edges, double *ww, int E, double *kin, double *kout, double *sin, double *sout, int N)
{
  double kmax=0.0;
  int i, n1, n2;
  for(i=1;i<=N;i++) kin[i]=kout[i]=sin[i]=sout[i]=0.0;


  for(i=1;i<=E;i++)
    {
      n1=edges[i][0];
      n2=edges[i][1];

      kin[n1]+=1.0;
      kout[n1]+=1.0;

      kin[n2]+=1.0;
      kout[n2]+=1.0;

      sin[n1]+=ww[i];
      sout[n1]+=ww[i];

      sin[n2]+=ww[i];
      sout[n2]+=ww[i];
    }
  for(i=1;i<=N;i++) if(kin[i]>kmax) kmax=kin[i];
  for(i=1;i<=N;i++) if(kout[i]>kmax) kmax=kout[i];
  return kmax;
}


double directed_stats(int **edges, double *ww, int E, double *kin, double *kout, double *sin, double *sout, int N)
{
  double kmax=0.0;
  int i, n1, n2;
  for(i=1;i<=N;i++) kin[i]=kout[i]=sin[i]=sout[i]=0.0;
  for(i=1;i<=E;i++)
    {
      n1=edges[i][0];
      n2=edges[i][1];

      //kin[n1]+=1.0;
      kout[n1]+=1.0;

      kin[n2]+=1.0;
      //kout[n2]+=1.0;

      //sin[n1]+=ww[i];
      sout[n1]+=ww[i];

      sin[n2]+=ww[i];
      //sout[n2]+=ww[i];
    }
  for(i=1;i<=N;i++) if(kin[i]>kmax) kmax=kin[i];
  for(i=1;i<=N;i++) if(kout[i]>kmax) kmax=kout[i];
  return kmax;
}





int optimalQ (int E, double *ww, int N, double *sin, double *sout)
{
  int i;
  double wmax=0.0, wmin=1e50;
  for(i=1;i<=E;i++)
    {
      if(ww[i]>wmax) wmax=ww[i];
      if(ww[i]<wmin) wmin=ww[i];
    }

  double smin=0.0, smax=0.0;
 
  for(i=1;i<=N;i++)
    {
      if(sin[i]>smax) smax=sin[i];
      if(sout[i]>smax) smax=sout[i];
    }
  smax+=10.0;


  double bestQ=log(smax/wmin)/log(2.0);
  printf("## Best Q %g\n",bestQ);

  return (int)(bestQ)+1;
}


double calculate_distr_weights(int E, double *ww, double *xx, double *prob, int Q, double kmax, double *sin, double *sout, int N)
{
  
  int i, k;
  double num_points=pow(2.0,Q);
  double *rr;
  rr=(double *)malloc(num_points*sizeof(double));
  for(i=0;i<num_points;i++) rr[i]=xx[i]=prob[i]=0.0;
  double wmax=0.0, wmin=1e50;
  for(i=1;i<=E;i++)
    {
      if(ww[i]>wmax) wmax=ww[i];
      if(ww[i]<wmin) wmin=ww[i];
    }
 

  double smin=0.0, smax=kmax*wmax;
  smax=0.0;
  for(i=1;i<=N;i++)
    {
      if(sin[i]>smax) smax=sin[i];
      if(sout[i]>smax) smax=sout[i];
    }
  smax+=10.0;
  
  printf("## minw %g maxw %g   mins %g maxs %g\n",wmin,wmax,wmin,smax);
  
 

  
  

  double dx=(smax-smin)/(double)num_points;
  xx[0]=smin;
  for(i=1;i<num_points;i++) xx[i]=xx[i-1]+dx;
  for(i=1;i<=E;i++)
    {
      for(k=0;k<num_points-1;k++)
    {
      if(ww[i]>=xx[k] && ww[i]<=xx[k+1]) goto next;
      
    }
    next:
      if(k==num_points-1) printf("%g\n",ww[i]);
      prob[k]+=1.0;
      rr[k]+=ww[i];
    }
  for(i=0;i<num_points;i++) if(prob[i]>0.0) rr[i]/=prob[i];
  for(i=0;i<num_points;i++) prob[i]/=((double)E*dx);
  //for(i=0;i<num_points;i++) xx[i]=rr[i];
  
  //for(i=0;i<num_points;i++) if(prob[i]>0.0) printf("%g %g %g\n",rr[i],xx[i],prob[i]);
  free(rr);

  return dx;
}





void print_table(char *filename, double **table, double *xx, int k, int N)
{
  FILE *f;
  f=fopen(filename,"w");
  int i;
  for(i=0;i<N;i++) fprintf(f,"%g %g\n",xx[i],table[k-1][i]);
  fclose(f);
}



void print_rank(char *filedata, int **edges, double *ww, double *pp, int E)
{
  FILE *f;
  int i;
  char filename[100];

  /*
  sprintf(filename,"DATA/rank_%s",filedata);
  f=fopen(filename,"w");
  for(i=1;i<=E;i++) fprintf(f,"%d %d %g\n",edges[i][0],edges[i][1],pp[i]);
  fclose(f);
  */

  sprintf(filename,"GloSS_%s",filedata);
  f=fopen(filename,"w");
  for(i=1;i<=E;i++) fprintf(f,"%d %d %g %g\n",edges[i][0],edges[i][1],pp[i],ww[i]);
  fclose(f);

}




