%% Load all the data (always run this section first)
clear all
close all
user = getenv('USER');
if ispc
    path=fullfile('C:','Users','battaallah','Documents','GitHub','CircleQuest_study01');
else
    path = fullfile('/Users',user,'Documents','GitHub','CircleQuest_study01');
end
addpath(genpath(fullfile(path, 'fcn')));
cd(path);


% Select all participants...
allfiles = dir(fullfile(path, 'data','raw','*.mat'));
for f = 1:size(allfiles,1)
    subjNames{f}=extractBefore(extractAfter(allfiles(f).name,9),11);
end
subjNames = unique(subjNames);
iexclude = find(strcmp(subjNames,'CQ01_YHC02') | strcmp(subjNames,'CQ01_YHC07') | strcmp(subjNames,'CQ01_YHC17'));
subjNames(iexclude)=[]


% Load questionnaires
nQuestions.AMI = 18;
nQuestions.AES = 14;
nQuestions.IUS = 27;
nQuestions.BIS = 30;
nQuestions.BDI = 21;
nQuestions.MCQ30 = 30;
clear temp
questionnaires = {'AMI', 'AES', 'IUS', 'BIS', 'BDI', 'MCQ30', 'Raven'}; % Enter here the questionnaires you want to load
allquestions=[];
for i = 1:length(questionnaires)
    for f = 1:length(subjNames)
        qdata=importQuestionnaire(fullfile(path,'data','questionnaires',[questionnaires{i} '*.csv']),questionnaires{i});
        try temp(f,:) = qdata(qdata.participant==subjNames{f},:);
        catch
            temp(f,:) = array2table(NaN(1,size(qdata,2)));
        end
    end
    quest.(questionnaires{i}) = temp;
    quest.(questionnaires{i}){(ismissing(temp.participant)),:}=missing; % Replace missing participants by a row of NaNs
    clear temp qdata
    
    if i<=6
        for q = 1:nQuestions.(questionnaires{i})
                allquestions = horzcat(allquestions,...
                    quest.(questionnaires{i}).(sprintf('Q%.2d',q)));
        end
    else
        allquestions = horzcat(allquestions,quest.Raven.total);
    end
end
 ap = quest.AMI.total>nanmedian(quest.AMI.total);

extra=importDemographics(fullfile(path,'data','demographics.xlsx'));
extra(~ismember(extra.participant,subjNames),:)=[];



%% QUESTIONNAIRES: correlation matrix
clear temp
for i = 1:length(questionnaires)
    temp{i}=quest.(questionnaires{i}).total;
end
demographics=table(temp{:},'VariableNames',questionnaires);
demographics.AMI = demographics.AMI./18;
demographics.AMI_beh = quest.AMI.beh./6;
demographics.AMI_social = quest.AMI.social./6;
demographics.AMI_emo = quest.AMI.emo./6;
[coeff,score,latent,tsquared,explained,mu]=pca(nanzscore([quest.AES.total,quest.AMI.total]),'NumComponents',1);
demographics.ApathyScore = score;
demographics.age = extra.age;
demographics.yearsOfEducation = extra.yoe;

% mycorrplot_1(totalScores)
figure
[R,PValue,H] = corrplot(demographics,'type','Pearson','testR','on','rows','pairwise');
demographics.groupAlloc = ap;

% Demographics table
clear m s M S t
m=varfun(@nanmean,demographics,'GroupingVariable',{'groupAlloc'});
s=varfun(@nanstd,demographics,'GroupingVariable',{'groupAlloc'});

[h p]=ttest2(table2array(demographics(ap,1:end-1)),table2array(demographics(~ap,1:end-1)));

M = table2array(m);
S = table2array(s);

t = array2table(horzcat(M(1,:)',S(1,:)',M(2,:)',S(2,:)',[NaN(2,1);p']));
t.Properties.VariableNames = {'lessApathetic_mean','lessApathetic_SD','moreApathetic_mean','moreApathetic_SD','Pvalue'};
t=t([2,14,3,10,11,12,4,13,6,7,5,8,9,15],:);
t.Properties.RowNames = {'n','age','AMI','AMI_beh','AMI_soc','AMI_emo','AES','ApathyScore',...
    'BIS','BDI','IUS','MCQ30','Raven','YearsOfEducation'};

%% Re-order rows and export table
writetable(t,...
    fullfile(path,'outputs','demographicTable.xls'),...
    'WriteRowNames',true);
figure
scatterRegress(demographics.age,demographics.AMI_beh);
xlabel('Age (years)');
ylabel('AMI behavioural score')










