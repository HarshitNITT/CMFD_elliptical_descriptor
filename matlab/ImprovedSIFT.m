%% Important Variables
% a : Input image
% kpmag : keypoints magnitude
% kpori : keypoints orientation
% kpd   : key point descriptors
% kp    : keypoints
% kpl   : keypoint locations

clc;
close all;
clear all;
image=input('Choose and enter the image name        :      ','s');
a=imread(image);
original=a;
imshow(a);
title('Selected image');
[m,n,plane]=size(a);
origin_m=m;
origin_n=n;
size(a)
if plane==3
a=rgb2gray(a);

end
a=im2double(a);
store1=[];
store2=[];
store3=[];
tic
%% 1st octave generation
k2=0;
[m,n]=size(a);
a(m:m+6,n:n+6)=0;
clear c;
for k1=0:3
    k=sqrt(2);
sigma=(k^(k1+(2*k2)))*1.6;
for x=-3:3
    for y=-3:3
        h(x+4,y+4)=(1/((2*pi)*((k*sigma)*(k*sigma))))*exp(-((x*x)+(y*y))/(2*(k*k)*(sigma*sigma)));
    end
end
for i=1:m
    for j=1:n
        t=a(i:i+6,j:j+6)'.*h;
        c(i,j)=sum(sum(t));
    end
end
store1=[store1 c];
end
clear a;
a=imresize(original,1/2);
a=imresize(original,[origin_m,origin_n]);
[m,n,plane]=size(a);
size(a)
if plane==3
a=rgb2gray(a);

end
a=im2double(a);

%% 2nd level pyramid generation
k2=1;
[m,n]=size(a);
a(m:m+6,n:n+6)=0;
clear c;
for k1=0:3
    k=sqrt(2);
sigma=(k^(k1+(2*k2)))*1.6;
for x=-3:3
    for y=-3:3
        h(x+4,y+4)=(1/((2*pi)*((k*sigma)*(k*sigma))))*exp(-((x*x)+(y*y))/(2*(k*k)*(sigma*sigma)));
    end
end
for i=1:m
    for j=1:n
        t=a(i:i+6,j:j+6)'.*h;
        c(i,j)=sum(sum(t));
    end
end
store2=[store2 c];
end
clear a;
a=imresize(original,1/4);
a=imresize(original,[origin_m,origin_n]);
[m,n,plane]=size(a);
size(a)
if plane==3
a=rgb2gray(a);

end
a=im2double(a);
%% 3rd level pyramid generation
k2=2;
[m,n]=size(a);
a(m:m+6,n:n+6)=0;
clear c;
for k1=0:3
    k=sqrt(2);
sigma=(k^(k1+(2*k2)))*1.6;
for x=-3:3
    for y=-3:3
        h(x+4,y+4)=(1/((2*pi)*((k*sigma)*(k*sigma))))*exp(-((x*x)+(y*y))/(2*(k*k)*(sigma*sigma)));
    end
end
for i=1:m
    for j=1:n
        t=a(i:i+6,j:j+6)'.*h;
        c(i,j)=sum(sum(t));
    end
end
store3=[store3 c];
end

%% Obtaining key point from the image
row=m;
col=n;
thres=0;
i1=store1(1:row,1:col)-store1(1:row,col+1:2*col);
i2=store1(1:row,col+1:2*col)-store1(1:row,2*col+1:3*col);
i3=store1(1:row,2*col+1:3*col)-store1(1:row,3*col+1:4*col);
[m,n]=size(i2);
    itemp=i2;

    [m,n]=size(i2);
    kp=[];
    kpl=[];
    tic
    for i=2:m-1
        for j=2:n-1
            x=i1(i-1:i+1,j-1:j+1);
            y=i2(i-1:i+1,j-1:j+1);
            z=i3(i-1:i+1,j-1:j+1);
            y(1:4)=y(1:4);
            y(5:8)=y(6:9);
            mx=max(max(x));
            mz=max(max(z));
            mix=min(min(x));
            miz=min(min(z));
            my=max(max(y));
            miy=min(min(y));
            if (i2(i,j)>my && i2(i,j)>mz ) || (i2(i,j)<miy && i2(i,j)<miz)
                kp=[kp i2(i,j)];
                kpl=[kpl i j];
            end
        end
    end
    fprintf('\nTime taken for finding the key points is :%f\n',toc);
    for i=1:m-1
        for j=1:n-1
             mag(i,j)=sqrt(((i2(i+1,j)-i2(i,j))^2)+((i2(i,j+1)-i2(i,j))^2));
             oric(i,j)=atan2(((i2(i+1,j)-i2(i,j))),(i2(i,j+1)-i2(i,j)))*(180/pi);
        end
    end

    kpmag=[];
    kpori=[];

    for x1=1:2:length(kpl)
        k1=kpl(x1);
        j1=kpl(x1+1);
        if k1 > 2 && j1 > 2 && k1 < m-2 && j1 < n-2
            p1=mag(k1-2:k1+2,j1-2:j1+2);
            q1=oric(k1-2:k1+2,j1-2:j1+2);
        else
            continue;
        end
        %% Finding orientation and magnitude for the key point
        [m1,n1]=size(p1);
        magcounts=[];
        for x=0:10:359
            magcount=0;
            for i=1:m1
                for j=1:n1
                    ch1=-180+x;
                    ch2=-171+x;
                     %if ch1<0  ||  ch2<0
                    if (q1(i,j))>=(ch1) && (q1(i,j))<(ch2)
                        %if abs(q1(i,j))<abs(ch1) && abs(q1(i,j))>=abs(ch2)

                        ori(i,j)=(ch1+ch2+1)/2;
                        magcount=magcount+p1(i,j);
                    end
                    % else
                    %     if abs(q1(i,j))>abs(ch1) && abs(q1(i,j))<=abs(ch2)
                    %         ori(i,j)=(ch1+ch2+1)/2;
                    %         magcount=magcount+p1(i,j);
                    %     end
                    % end
                end
            end
            magcounts=[magcounts magcount];
        end
        [maxvm maxvp]=max(magcounts);
        kmag=maxvm;
        kori=(((maxvp*10)+((maxvp-1)*10))/2)-180;
        kpmag=[kpmag kmag k1 j1];
        kpori=[kpori kori];

    end
counter=0;
for t =0:3
    kend=[];
    for x1=1:3:length(kpmag)
       
        if kpmag(x1) >= thres
            kend=[kend kpmag(x1+1) kpmag(x1+2)];
        end    
        
       
            
        
    end    
        
    %finding the KUM value
    for i=1:m
        for j=1:n
            si(i,j)=0;
        end
    end
    for i=1:2:length(kend);
        si(ceil(kend(i)/6),ceil(kend(i+1)/6))= si(ceil(kend(i)/6),ceil(kend(i+1)/6))+1;
        
    end
    no_of_key_points=length(kend)/2;
    key_points_density=no_of_key_points/(ceil(n/6)*ceil(m/6));
    kum_square=0;
    for i=1:ceil(m/6)
        for j=1:ceil(n/6)
            kum_square=kum_square+(key_points_density-si(i,j))*(key_points_density-si(i,j));
        end
    end
    kum=sqrt(kum_square/no_of_key_points);
    thres=thres/2;
    if kum <0.3
        break;

    end
end

    clear a;
    a=imresize(original,1/((k2+1)*2));
    

%% 2nd Octave generation
% k2=1;
% [m,n]=size(a);
% a(m:m+6,n:n+6)=0;
% clear c;
% for k1=0:3
%     k=sqrt(2);
% sigma=(k^(k1+(2*k2)))*1.6;
% for x=-3:3
%     for y=-3:3
%         h(x+4,y+4)=(1/((2*pi)*((k*sigma)*(k*sigma))))*exp(-((x*x)+(y*y))/(2*(k*k)*(sigma*sigma)));
%     end
% end
% for i=1:m
%     for j=1:n
%         t=a(i:i+6,j:j+6)'.*h;
%         c(i,j)=sum(sum(t));
%     end
% end
% store2=[store2 c];
% end
% clear a;
% a=imresize(original,1/((k2+1)*2));

% %% 3rd octave generation
% k2=2;
% [m,n]=size(a);
% a(m:m+6,n:n+6)=0;
% clear c;
% for k1=0:3
%     k=sqrt(2);
% sigma=(k^(k1+(2*k2)))*1.6;
% for x=-3:3
%     for y=-3:3
%         h(x+4,y+4)=(1/((2*pi)*((k*sigma)*(k*sigma))))*exp(-((x*x)+(y*y))/(2*(k*k)*(sigma*sigma)));
%     end
% end
% for i=1:m
%     for j=1:n
%         t=a(i:i+6,j:j+6)'.*h;
%         c(i,j)=sum(sum(t));
%     end
% end
% store3=[store3 c];
% end
% [m,n]=size(original);
% fprintf('\nTime taken for Pyramid level generation is :%f\n',toc);

% %% Obtaining key point from the image
% i1=store1(1:m,1:n)-store1(1:m,n+1:2*n);
% i2=store1(1:m,n+1:2*n)-store1(1:m,2*n+1:3*n);
% i3=store1(1:m,2*n+1:3*n)-store1(1:m,3*n+1:4*n);


% [m,n]=size(i2);
% kp=[];
% kpl=[];
 tic
% for i=2:m-1
%     for j=2:n-1
%         x=i1(i-1:i+1,j-1:j+1);
%         y=i2(i-1:i+1,j-1:j+1);
%         z=i3(i-1:i+1,j-1:j+1);
%         y(1:4)=y(1:4);
%         y(5:8)=y(6:9);
%         mx=max(max(x));
%         mz=max(max(z));
%         mix=min(min(x));
%         miz=min(min(z));
%         my=max(max(y));
%         miy=min(min(y));
%         if (i2(i,j)>my && i2(i,j)>mz) || (i2(i,j)<miy && i2(i,j)<miz)
%             kp=[kp i2(i,j)];
%             kpl=[kpl i j];
%         end
%     end
% end
fprintf('\nTime taken for finding the key points is :%f\n',toc);

%% Finding Image Derivatives
%[Gx,Gy] = imgradientxy(i2,'central')
%[Gxx,Gxy]=imgradientxy(Gx,'central')
%[Gyx,Gyy]=imgradientxy(Gy,'central')

%% Key points plotting on to the image
for it=1:2:length(kpl);
    k1=kpl(it);
    j1=kpl(it+1);
    moment_matrix(1:2,1:2)=0;
    for j=-4:4;
        for i=-floor(sqrt(1-(j*j)/16)):ceil(sqrt(1-(j*j)/16));
            x=j;
            y=i;
            k=2;
            gauss_kernel=(1/((2*pi)*((k*sigma)*(k*sigma))))*exp(-((x*x)+(y*y))/(2*(k*k)*(sigma*sigma)));
            if(i+j1>1 && i+j1+1<n && j+k1>1 && j+1+k1<m)
                x_dir=(i2(j+1+k1,i+j1)-i2(j+k1-1,i+j1))*(i2(j+1+k1,i+j1)-i2(j+k1-1,i+j1));
                y_dir=(i2(j+k1,i+j1+1)-i2(j+k1,i+j1-1))*(i2(j+k1,i+j1+1)-i2(j+k1,i+j1-1));
                xy_dir=(i2(j+1+k1,i+j1)-i2(j+1+k1,i+j1))*(i2(j+k1,i+j1+1)-i2(j+k1,i+j1-1));
                moment_matrix(1,1)=moment_matrix(1,1)+x_dir;
                moment_matrix(1,2)=moment_matrix(1,2)+xy_dir;
                moment_matrix(2,1)=moment_matrix(2,1)+xy_dir;
                moment_matrix(2,2)=moment_matrix(2,2)+y_dir;
                    
                    
                    
            end
        end
    end
    sqrtm(moment_matrix);
    trans=(transpose(sqrtm(moment_matrix)));
    
    % for ii=-4:4;
    %     for ij=-floor(sqrt(1-(j*j)/16)):ceil(sqrt(1-(j*j)/16));;
    %         end_res=[ii ij]*(trans);
    %         inew=ceil(end_res(1));
    %         jnew=ceil(end_res(2));
    %         if ii==0 && ij==0
    %             kpl(it)=(ceil(inew+k1));
    %             kpl(it+1)=(ceil(jnew+j1));
    %         end
    %         if inew+k1>0 && jnew+j1>0 && inew+k1<=m && jnew+j1<=n && ii+k1>0 && ii+k1<=m && ij+j1>0 && ij+j1<=n 
    %         i2(ceil(inew+k1),ceil(jnew+j1))=itemp(ii+k1,ij+j1); 
    %         end
    %     end
    % end
   
    i2(k1,j1)=1;
end

figure, imshow(itemp);
title('Image with key points mapped onto it');
figure, imshow(i2);
%%
% for i=1:m-1
%     for j=1:n-1
%          mag(i,j)=sqrt(((i2(i+1,j)-i2(i,j))^2)+((i2(i,j+1)-i2(i,j))^2));
%          oric(i,j)=atan2(((i2(i+1,j)-i2(i,j))),(i2(i,j+1)-i2(i,j)))*(180/pi);
%     end
% end

% %% Forming key point neighbourhooods
% kpmag=[];
% kpori=[];
% for x1=1:2:length(kpl)
%     k1=kpl(x1);
%     j1=kpl(x1+1);
%     if k1 > 2 && j1 > 2 && k1 < m-2 && j1 < n-2
%         p1=mag(k1-2:k1+2,j1-2:j1+2);
%         q1=oric(k1-2:k1+2,j1-2:j1+2);
%     else
%         continue;
%     end
%     %% Finding orientation and magnitude for the key point
% [m1,n1]=size(p1);
% magcounts=[];
% adi=[];
% count=0;
% for x=0:10:359
%     magcount=0;
    
% for i=1:m1
%     for j=1:n1
%         ch1=-180+x;
%         ch2=-171+x;
%         %if ch1<0  ||  ch2<0
%             if (q1(i,j))>=(ch1) && (q1(i,j))<(ch2)
%                 %if abs(q1(i,j))<abs(ch1) && abs(q1(i,j))>=abs(ch2)

%                 ori(i,j)=(ch1+ch2+1)/2;
%                 magcount=magcount+p1(i,j);
%             end
%         % else
%         %     if abs(q1(i,j))>abs(ch1) && abs(q1(i,j))<=abs(ch2)
%         %         ori(i,j)=(ch1+ch2+1)/2;
%         %         magcount=magcount+p1(i,j);
%         %     end
%         % end
%     end
% end
% count=count+1;
% magcounts=[magcounts magcount];
% adi=[adi magcount];
% end
% [maxvm maxvp]=max(magcounts);
% kmag=maxvm;
% kori=(((maxvp*10)+((maxvp-1)*10))/2)-180;
% kpmag=[kpmag kmag];
% kpori=[kpori kori];
% size(kpmag)
% size(kp)
% % maxstore=[];
% % for i=1:length(magcounts)
% %     if magcounts(i)>=0.8*maxvm
% %         maxstore=[maxstore magcounts(i) i];
% %     end
% % end
% % 
% % if maxstore > 2
% %     kmag=maxstore(1:2:length(maxstore));
% %     maxvp1=maxstore(2:2:length(maxstore));
% %     temp=(countl((2*maxvp1)-1)+countl(2*maxvp1)+1)/2;
% %     kori=temp;
% % end
% end
% fprintf('\nTime taken for magnitude and orientation assignment is :%f\n',toc);


%% Forming key point Descriptors
kpd=[];
kpd1=[];
for x1=1:2:length(kend);
    k1=kend(x1);
    j1=kend(x1+1);
   
    kpmagd=[];
    kporid=[];
    res=[];
    r=[];
    c = zeros(1,128);
    for x=0:45:359
        magcount=0;
        for i=-4:4;
            for j=-floor(sqrt(16-i*i)):floor(sqrt(16-i*i));
                
                ch1=-180+x;
                ch2=-180+45+x;
                %if ch1<0  ||  ch2<0
                    if (k1+i)>0 && (j1+j)>0 && (k1+i)<m && (j1+j)<n
                        
                        %if abs(oric(k1+i,j1+j))<abs(ch1) && abs(oric(k1+i,j1+j))>=abs(ch2)
                        if (oric(k1+i,j1+j))>=(ch1) && oric(k1+i,j1+j)<(ch2)
                           
                            if i<=0 && j<=0
                                
                                if abs(i)>abs(j)
                                %y=2 finding x
                            
                                    if i*i+j*j<=1
                                        c(2*8+floor(x/45)+1)=c(2*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                    elseif i*i+j*j <=4
                                        c(6*8+floor(x/45)+1)=c(6*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                    elseif i*i+j*j <=9 
                                        c(10*8+floor(x/45)+1)=c(10*8+floor(x/45)+1)+mag(k1+i,j1+j);
                               
                                    
                                    elseif i*i+j*j <=16 
                                        
                                        c(14*8+floor(x/45)+1)=c(14*8+floor(x/45)+1)+mag(k1+i,j1+j);   
                                    
                                    
                                    end
                                

                                else
                                %y=1 finding x
                                    if i*i+j*j <=1
                                        c(1*8+floor(x/45)+1)=c(1*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                    
                                    elseif i*i+j*j <=4
                                        c(5*8+floor(x/45)+1)=c(5*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                   
                                    
                                    elseif i*i+j*j <=9 
                                        c(9*8+floor(x/45)+1)=c(9*8+floor(x/45)+1)+mag(k1+i,j1+j);
                               
                                    
                                    elseif i*i+j*j<=16
                                        c(13*8+floor(x/45)+1)=c(13*8+floor(x/45)+1)+mag(k1+i,j1+j);   
                                    end
                                end
                           
                            end

                            if i>=0 && j<=0
                                if abs(i)>=abs(j)
                               %y=0 finding x
                                    if i*i+j*j<=1
                                        
                                        c(0*8+floor(x/45)+1)=c(0*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                    
                                    elseif i*i+j*j <=4
                                        c(4*8+floor(x/45)+1)=c(4*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                   
                                    
                                    elseif i*i+j*j <= 9
                                        c(8*8+floor(x/45)+1)=c(8*8+floor(x/45)+1)+mag(k1+i,j1+j);
                           
                                
                                    elseif i*i+j*j <=16
                                    c(12*8+floor(x/45)+1)=c(12*8+floor(x/45)+1)+mag(k1+i,j1+j);   
                                    end
                           
                                else
                                %y= 1
                                    if i*i+j*j<=1
                                        c(1*8+floor(x/45)+1)=c(1*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                    
                                    elseif i*i+j*j <=4
                                        c(5*8+floor(x/45)+1)=c(5*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                   
                                    
                                    elseif i*i+j*j <= 9
                                        c(9*8+floor(x/45)+1)=c(9*8+floor(x/45)+1)+mag(k1+i,j1+j);
                               
                                
                                    elseif i*i+j*j <=16
                                        c(13*8+floor(x/45)+1)=c(13*8+floor(x/45)+1)+mag(k1+i,j1+j);   
                                    end
                                end
                            end
                        
                            if i>=0 && j>=0
                                if abs(i)>abs(j)
                               %0 
                                    if i*i+j*j<=1
                                       k1+i
                                        c(0*8+floor(x/45)+1)=c(0*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                    
                                    elseif i*i+j*j <=4
                                        c(4*8+floor(x/45)+1)=c(4*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                   
                                    
                                    elseif i*i+j*j <= 9
                                        c(8*8+floor(x/45)+1)=c(8*8+floor(x/45)+1)+mag(k1+i,j1+j);
                               
                                    
                                    elseif i*i+j*j <=16
                                        c(12*8+floor(x/45)+1)=c(12*8+floor(x/45)+1)+mag(k1+i,j1+j);   
                                    end

                                else
                              %3
                                    if i*i+j*j <=1
                                        c(3*8+floor(x/45)+1)=c(3*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                    
                                    elseif i*i+j*j <=4
                                        c(7*8+floor(x/45)+1)=c(7*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                   
                                    
                                    elseif i*i+j*j <=9 
                                        c(11*8+floor(x/45)+1)=c(11*8+floor(x/45)+1)+mag(k1+i,j1+j);
                               
                                    
                                    elseif i*i+j*j<=16
                                        c(15*8+floor(x/45)+1)=c(15*8+floor(x/45)+1)+mag(k1+i,j1+j);   
                                    end 
                                end
                            end
                        
                            if i<=0 && j>=0
                                if abs(i)>abs(j)
                                %2
                                    if i*i+j*j<=1
                                        c(2*8+floor(x/45)+1)=c(2*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                   
                                    elseif i*i+j*j <=4
                                        c(6*8+floor(x/45)+1)=c(6*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                   
                                    
                                    elseif i*i+j*j <=9 
                                        c(10*8+floor(x/45)+1)=c(10*8+floor(x/45)+1)+mag(k1+i,j1+j);
                               
                                    
                                    elseif i*i+j*j <=16 
                                        c(14*8+floor(x/45)+1)=c(14*8+floor(x/45)+1)+mag(k1+i,j1+j);   
                                    end
                                
                                else
                                %3
                                    if i*i+j*j<=1
                                        c(3*8+floor(x/45)+1)=c(3*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                    
                                    elseif i*i+j*j <=4
                                        c(7*8+floor(x/45)+1)=c(7*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                   
                                    
                                    elseif i*i+j*j <= 9
                                        c(11*8+floor(x/45)+1)=c(11*8+floor(x/45)+1)+mag(k1+i,j1+j);
                               
                                    
                                    elseif i*i+j*j <=16
                                        c(15*8+floor(x/45)+1)=c(15*8+floor(x/45)+1)+mag(k1+i,j1+j);   
                                    end
                                end
                            end
                        end
                    end
                
            end            
        end  
    end
    kpd1=[kpd1 c];
    c = zeros(1,128);
    for x=0:45:359
        magcount=0;
        for i=-4:4;
            for j=-floor(sqrt(16-i*i)):floor(sqrt(16-i*i));
                
                ch1=-180+x;
                ch2=-180+45+x;
                %if ch1<0  ||  ch2<0
                    if (k1+i)>0 && (j1+j)>0 && (k1+i)<m && (j1+j)<n
                        
                        %if abs(oric(k1+i,j1+j))<abs(ch1) && abs(oric(k1+i,j1+j))>=abs(ch2)
                        if (oric(k1+i,j1+j))>=(ch1) && oric(k1+i,j1+j)<(ch2)
                           
                            if i<=0 && j<=0
                                
  
                                %y=1 finding x
                            
                                    if i*i+j*j<=1
                                        c(1*8+floor(x/45)+1)=c(1*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                    elseif i*i+j*j <=4
                                        c(5*8+floor(x/45)+1)=c(5*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                    elseif i*i+j*j <=9 
                                        c(9*8+floor(x/45)+1)=c(9*8+floor(x/45)+1)+mag(k1+i,j1+j);
                               
                                    
                                    elseif i*i+j*j <=16 
                                        
                                        c(13*8+floor(x/45)+1)=c(13*8+floor(x/45)+1)+mag(k1+i,j1+j);   
                                    
                                    
                                    end
                            end

                            if i>=0 && j<=0
                               %y=0 finding x
                                    if i*i+j*j<=1
                                        
                                        c(0*8+floor(x/45)+1)=c(0*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                    
                                    elseif i*i+j*j <=4
                                        c(4*8+floor(x/45)+1)=c(4*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                   
                                    
                                    elseif i*i+j*j <= 9
                                        c(8*8+floor(x/45)+1)=c(8*8+floor(x/45)+1)+mag(k1+i,j1+j);
                           
                                
                                    elseif i*i+j*j <=16
                                    c(12*8+floor(x/45)+1)=c(12*8+floor(x/45)+1)+mag(k1+i,j1+j);   
                                    end
                           
                                
                            end
                        
                            if i>=0 && j>=0
                                
                                 
                              %3
                                    if i*i+j*j <=1
                                        c(3*8+floor(x/45)+1)=c(3*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                    
                                    elseif i*i+j*j <=4
                                        c(7*8+floor(x/45)+1)=c(7*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                   
                                    
                                    elseif i*i+j*j <=9 
                                        c(11*8+floor(x/45)+1)=c(11*8+floor(x/45)+1)+mag(k1+i,j1+j);
                               
                                    
                                    elseif i*i+j*j<=16
                                        c(15*8+floor(x/45)+1)=c(15*8+floor(x/45)+1)+mag(k1+i,j1+j);   
                                    end 
                            end
                            
                        
                            if i<=0 && j>=0

                                %2
                                    if i*i+j*j<=1
                                        c(2*8+floor(x/45)+1)=c(2*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                   
                                    elseif i*i+j*j <=4
                                        c(6*8+floor(x/45)+1)=c(6*8+floor(x/45)+1)+mag(k1+i,j1+j);
                                   
                                    
                                    elseif i*i+j*j <=9 
                                        c(10*8+floor(x/45)+1)=c(10*8+floor(x/45)+1)+mag(k1+i,j1+j);
                               
                                    
                                    elseif i*i+j*j <=16 
                                        c(14*8+floor(x/45)+1)=c(14*8+floor(x/45)+1)+mag(k1+i,j1+j);   
                                    end
                                
                                
                            end
                        end
                    end
                
            end            
        end  
    end           
    kpd=[kpd c];          
                
end          
%%
final_matching=[];

norm_fin=[];
rot90_fin=[];
rot180_fin=[];
rot270_fin=[];
rotsec90_fin=[];
rotsec180_fin=[];
rotsec270_fin=[];
rot315=[];
thres=0.9
for t=1:10
    flag=0;
for i=1:length(kend)/2;
    dist_norm=[];
    rot90=[];
    rot180=[];
    rot270=[];
    rotsec90=[];
    rotsec180=[];
    rotsec270=[];
    position_norm=[];
    position_rot90=[];
    position_rot180=[];
    position_rot270=[];
    dist_hor=[];
    dist_vert=[];
    position_rot90_x=[];
    position_rot90_y=[];
    position_rot180_x=[];
    position_rot180_y=[];
    position_rot270_x=[];
    position_rot270_y=[];
    position_rotsec90_x=[];
    position_rotsec90_y=[];
    position_rotsec180_x=[];
    position_rotsec180_y=[];
    position_rotsec270_x=[];
    position_rotsec270_y=[];
    position_norm_x=[];
    position_norm_y=[];
   
    for j=1:length(kend)/2;
        if i==j
            continue;
        end
        normal=0;
    current=(i-1)*128;
    matched=(j-1)*128;
    
    for diff=1:3;
        s1=0;
        s2=0;
        for start_first=1:4;
            for k=1:32;
                s1=s1+(kpd1(current+(start_first-1)*32+k)-kpd1(matched+(mod((start_first+diff-1),4)*32+k)))^2;
                % s2=s2+(kpd1(current+(start_first-1)*32+k)-kpd1(matched+(mod((start_first-diff+3),4)*32+k)))^2;
            end
        end
        if diff==1

            rot90=[rot90 (s1)];
            position_rot90_x=[position_rot90_x kend(2*(j-1)+1)];
            position_rot90_y=[position_rot90_y kend(2*j)];
        end
        if diff==2
            
            rot180=[rot180 (s1)];
            position_rot180_x=[position_rot180_x kend(2*(j-1)+1)];
            position_rot180_y=[position_rot180_y kend(2*j)];
        end
        if diff==3
            position_rot270_x=[position_rot270_x kend(2*(j-1)+1)];
            position_rot270_y=[position_rot270_y kend(2*j)];
           rot270=[rot270 (s1)];
        end
    end



    for diff=1:3;
        s1=0;
        s2=0;
        for start_first=1:4;
            for k=1:32;
                s1=s1+(kpd(current+(start_first-1)*32+k)-kpd(matched+(mod((start_first+diff-1),4)*32+k)))^2;
                % s2=s2+(kpd(current+(start_first-1)*32+k)-kpd(matched+(mod((start_first-diff+3),4)*32+k)))^2;
            end
        end
        if diff==1
            rotsec90=[rotsec90 s1];
            position_rotsec90_x=[position_rotsec90_x kend(2*(j-1)+1)];
            position_rotsec90_y=[position_rotsec90_y kend(2*j)];
        end
        if diff==2
            rotsec180=[rotsec180 s1];
            position_rotsec180_x=[position_rotsec180_x kend(2*(j-1)+1)];
            position_rotsec180_y=[position_rotsec180_y kend(2*j)];

        end
        if diff==3
            rotsec270=[rotsec270 s1];
            position_rotsec270_x=[position_rotsec270_x kend(2*(j-1)+1)];
            position_rotsec270_y=[position_rotsec270_y kend(2*j)];
        end
    end


    for k=1:128
            normal=normal+(kpd1(current+k)-kpd1(matched+k))*(kpd1(current+k)-kpd1(matched+k));
    end
  
    dist_norm=[dist_norm sqrt(normal)];
    position_norm_x=[position_norm_x kend(2*(j-1)+1)];
    position_norm_y=[position_norm_y kend(2*j)];

    end
    [dist_norm,sortorder_norm]=sort(dist_norm,'descend');

    
    [rot90,sortorder_rot90]=sort(rot90,'descend');
    [rot180,sortorder_rot180]=sort(rot180,'descend');
    [rot270,sortorder_rot270]=sort(rot270,'descend');

    % [rotsec90,sortorder_rotsec90]=sort(rotsec90,'descend');
    % [rotsec180,sortorder_rotsec180]=sort(rotsec180,'descend');
    % [rotsec270,sortorder_rotsec270]=sort(rotsec270,'descend');
    
   
    ratio_norm=(dist_norm(2)/dist_norm(1));
 
    
     ratio_90=(rot90(2)/rot90(1));
    ratio_180=(rot180(2)/rot180(1));
    ratio_270=(rot270(2)/rot270(1));
  
    % ratiosec_90=(rotsec90(2)/rot90(1));
    % ratiosec_180=(rotsec180(2)/rot180(1));
    % ratiosec_270=(rotsec270(2)/rot270(1));
    
    norm_fin=[norm_fin ratio_norm];
rot90_fin=[rot90_fin ratio_90];
rot180_fin=[rot180_fin ratio_180];
rot270_fin=[rot270_fin ratio_270];
% rotsec90_fin=[rotsec90_fin ratiosec_90];
% rotsec180_fin=[rotsec180_fin ratiosec_180];
% rotsec270_fin=[rotsec270_fin ratiosec_270];

      xf=[];
    yf=[];
    x=[];
    y=[];

    if ratio_90<thres
               position_rot90_x=position_rot90_x(sortorder_rot90);
        position_rot90_y=position_rot90_y(sortorder_rot90);
        x=[x position_rot90_x(1)];
        y=[y position_rot90_y(1)];
    end
    if ratio_180<thres
                       position_rot180_x=position_rot180_x(sortorder_rot180);
        position_rot180_y=position_rot180_y(sortorder_rot180);
        x=[x position_rot180_x(1)];
        y=[y position_rot180_y(1)];
    end
    if ratio_270<thres
                       position_rot270_x=position_rot270_x(sortorder_rot270);
        position_rot270_y=position_rot270_y(sortorder_rot270);
        x=[x position_rot270_x(1)];
        y=[y position_rot270_y(1)];
    end

                       position_norm_x=position_norm_x(sortorder_norm);
        position_norm_y=position_norm_y(sortorder_norm);
        x=[x position_norm_x(1)];
        y=[y position_norm_y(1)];
   
  %   elseif ratiosec_90<thres
  %       position_rotsec90_x=position_rotsec90_x(sortorder_rotsec90);
  %       position_rotsec90_y=position_rotsec90_y(sortorder_rotsec90);
  %     x=[x position_rotsec90_x(1)];
  %     y=[y position_rotsec90_y(1)];
  %   elseif ratiosec_180<thres
  %             position_rotsec180_x=position_rotsec180_x(sortorder_rotsec180);
  %       position_rotsec180_y=position_rotsec180_y(sortorder_rotsec180);
        % x=[x position_rotsec90_x(1)];
  %     y=[y position_rotsec90_y(1)];
  %   elseif ratiosec_270<thres
  %              position_rotsec270_x=position_rotsec270_x(sortorder_rotsec270);
  %       position_rotsec270_y=position_rotsec270_y(sortorder_rotsec270);
  %     x=[x position_rotsec270_x(1)]
  %     y=[y position_rotsec270_y(1)]
  %   end
                if (ratio_norm<thres || ratio_90<thres || ratio_180<thres || ratio_270<thres )
                  if(length(final_matching)<0.3*length(kpl)) 
               final_matching=[final_matching kend(2*(i-1)+1) kend(2*i)  position_norm_x(1) position_norm_y(1)];
                  else
                      flag=1;
                  end
                  
    end

        
    
    end    
  if flag==0
      thres=thres+0.01
  else
      break;
  end

    
end
%% Dividing into 4x4 blocks

a=imread(image)
figure,imshow(a)
hold on 

for i=1:4:length(final_matching);
    plot([final_matching(i),final_matching(i+1)],[final_matching(i+2),final_matching(i+3)],'Color','g','LineWidth',1);
end
hold off
fprintf('\nTime taken for finding key point desctiptors is :%f\n',toc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%         END OF THE PROGRAM        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
