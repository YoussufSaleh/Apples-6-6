%% This script will determine the demographics for my study. 




%% QUESTIONNAIRES: correlation matrix


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
  

Chi = array2table(horzcat(FQs_Ex.GenderM1, FQs_Ex.groupAlloc));








