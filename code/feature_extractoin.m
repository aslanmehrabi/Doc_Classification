% Aslan Mehrabi  
% Document Classification of New York Times Articles
% Feature Extraction Methods: Bag of words / TF_IDF / Diemension Reduction


numLine=2000
%fclose (all)
fid = fopen('data_train.txt');
tline = fgetl(fid);

cnt=0;
%trainData=zeros(2000,1000);

maxLen=1400;
trainData=cell(2000,maxLen);
trainDataLen = zeros(2000,1);
while ischar(tline)
     
    cnt=cnt+1;
    %tmp=fgetl(fid);
    tmp = strread(tline, '%s', 'delimiter', ' ')';
    trainDataLen(cnt)=size(tmp,2);
    trainData(cnt,1:trainDataLen(cnt)) = tmp(1,:);
    
    tline = fgetl(fid);
    
end
fclose(fid);


dict ={};
for i = 1:numLine
    i
    tmp = strcat(trainData(i,1:trainDataLen(i)),{' '});
    str = [tmp{:}];
    dict = unique([dict, unique(regexp(str, ' ', 'split'))]);
end


numFeatures = size(dict,2);
trainFeatures = sparse(numLine,numFeatures);
for i =1:numLine
    i
    for j = 1:numFeatures
        trainFeatures(i,j) = sum(strcmp(trainData(i,1:trainDataLen(i)),dict(j)));
    end
    
    %{
    tmp = strcat(trainData(i,1:trainDataLen(i)),{' '});
    str = [tmp{:}];
    for j = 1:numFeatures
        temp=regexp(str, ' ', 'split');
        trainFeatures(i,j) = sum(strcmp(temp,dict(j)));
    end
    %}
    
    
end




% read the validatoin data

fid = fopen('data_valid.txt');
tline = fgetl(fid);

cnt=0;

maxLenValid=1400;
validData=cell(2000,maxLenValid);
validDataLen = zeros(2000,1);
maxx=0;
while ischar(tline)
     
    cnt=cnt+1;
    %tmp=fgetl(fid);
    tmp = strread(tline, '%s', 'delimiter', ' ')';
    validDataLen(cnt)=size(tmp,2);
    if(validDataLen(cnt) > maxx)
        maxx = validDataLen(cnt);
    end
    validData(cnt,1:validDataLen(cnt)) = tmp(1,:);
    
    tline = fgetl(fid);
    
end
fclose(fid);



numFeatures = size(dict,2);
validFeatures = sparse(numLine,numFeatures);
for i =1:numLine
    i
    for j = 1:numFeatures
        validFeatures(i,j) = sum(strcmp(validData(i,1:validDataLen(i)),dict(j)));
    end
    
    
    
end






tf_idf = sparse(numLine,numFeatures);

[x y]=find(trainFeatures);

len = size(x,1)
 
tf=0;
idf=0;

for i = 1:len
    i
   tf = trainFeatures(x(i),y(i)) / max(trainFeatures(x(i),:));
   idf = log(numLine / (sum(trainFeatures(:,y(i))) >= 1) );
   tf_idf(x(i),y(i)) = tf*idf;
end
   


fid = fopen('labels_train.txt');

getl = fgetl(fid);
trainLabel = zeros(numLine,1);

cnt=0;
while ischar(getl)
    cnt = cnt+1;
    trainLabel(cnt,1) = str2num(getl);
    getl = fgetl(fid);
end

fclose(fid);







% valid data



tf_idfValid = sparse(numLine,numFeatures);

[x y]=find(validFeatures);

len = size(x,1)
 
tf=0;
idf=0;

for i = 1:len
    i
   tf = validFeatures(x(i),y(i)) / max(validFeatures(x(i),:));
   idf = log(numLine / (sum(validFeatures(:,y(i))) >= 1) );
   tf_idfValid(x(i),y(i)) = tf*idf;
end
   



fid = fopen('labels_valid.txt');

getl = fgetl(fid);
validLabel = zeros(numLine,1);

cnt=0;
while ischar(getl)
    cnt = cnt+1;
    validLabel(cnt,1) = str2num(getl);
    getl = fgetl(fid);
end

fclose(fid);



%%%%%%




%{


words = {'do', 're', 'mi'};
definitions = {'a deer', 'a drop of golden sun', 'a name I call myself'};
 dictionary = createDictionary(words, definitions);

%dictionary for the words
dic = containers.Map
dic('foo') = 1
dic(' not a var name ') = 2
keys(dic)
values(dic)

%}

%fid = fopen('data_valid.txt')
%Out  = textscan(fid,'%s','delimiter',sprintf('\n'));
%fclose(fid)

%f = textread('data_train.txt','%s','delimiter','\n','whitespace','');
%read1= regexp(f{1},'\s','split')
%read = char(f)
%read2 = strread(read, '%s', 'delimiter', ' ')



    