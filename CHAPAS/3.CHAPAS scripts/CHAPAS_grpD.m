


clear
% STEP 1: Add all relevant files to the path 
%First MATLIB 
addpath(genpath('/Users/youssufsaleh/Documents/Master folder/Apples v2/matlib'));
% then the schizophrenia data
addpath(genpath('/Users/youssufsaleh/Dropbox (Neurological Conditions)/CHAPAS data (1)/Data'))
addpath(genpath('CHAPAS'));
% then the control data
addpath(genpath('/Users/youssufsaleh/Documents/Master folder/Apples v2/Raw-data'));
%first create group names. You can add multiple allStudyGroups but for this
%particular script I will stick with one. 

% Extract the relevant string components from both datasets so you can add
% them back into your for loop when it is time to load them all up. 






allStudyGroups = {{'Apples_%g', [100:139]},...
    }; %expects the raw data to have file format Apples_YC0*.mat

groupName = {'CHAPAS'};
% Move data into an array (one cell for on and off), each row = 1 subject
%loop through allStudyGroups
% create a numerical array that corresponds to our patient codes (100:132)

code = [100:139];
for subs = 1:length(allStudyGroups)
    alldata=[];
    %loop through subjects
    for i = 1:length(allStudyGroups{subs}{2})
        data = load(sprintf(allStudyGroups{subs}{1},code(i)));
        d=data.result.data; %retrieve trial data
        for t=126:135          %for the last 10 trials collecting force data
            AUC1{subs}(i,t) = nansum(d(t).data); % extract AUC data before removing data and 2 fields
            
            if d(t).Yestrial ==1 % if accepted
                if ~isnan(d(t).reward) % if data recorded
                    mF{subs}(i,t) = max(d(t).data(1:500));%500 is trial length
                else mF{subs}(i,t)=nan;
                end
            else mF{subs}(i,t)=nan;
            end
            
        end
        
        d=rmfield(d, 'data'); % don't process the actual squeeze data
        
        if isempty(alldata) % first subject: new structure
            alldata = d;
        else % subsequent subjects - make sure fields match
            d=ensureStructsAssignable(d, alldata);
            alldata=[alldata;d]; % and then add a new row
        end
    end
    d=transpIndex(alldata);
    alld{subs}=d;
    
    end
    
   
    
   D=alld{subs};
   


    save('AGT_DATA_CHAPAS40');

    save('mF','mF')
    save('AUC1','AUC1')


%%
if ~exist('D','var') % if above has not been run earlier load data that is already in folder
    load AGT_DATA_CHAPAS36 % or 'AGT_DATA_CHAPAS36-2' if you are including the excluded patients. 
end

alld=D;
filter = allStudyGroups{1}{2}'; % select appropriate patients
subj = length(filter);

load AUC1;
AUC1{1} = AUC1{1}(filter,:);


%% Create MVC matrix
for i = 1:length(allStudyGroups{1}{2}) % for each sub
    mvc{1}(i,1)=max(D.MVC(i,:,1)); %extract the max MVC for the subject (from 10 trials at the end)
end
% and now reshape data into arrays

for i=1:size(D.R,1) % for each subject in each state
    es =  [0.16 0.32 0.48 0.64 0.80]; %effort levels
    ss =  [1 4 7 10 13];%stake
    
    
    y = d.Yestrial(i,:)'; % did they accept?
    y = y(1:125); %MV we only need the first 125 trials as last 10 are not decisions
    y(isnan(y))=0; % compensate for someone's incompetence
    dt{subs} = d.endChoice-d.startChoice; %MV decision time
    dt{subs} = dt{subs}(:,1:125); %dt{subs}(:,37:end) %exclude practice session
    maxF = d.maximumForce(:,126:135); %MV we only need last 10 trials
    AUC_R = AUC1{subs}(:,126:135);%right % AUC
    % added by YS on 23/8/2018
    yesLoc = d.yeslocation(i,1:125)';
   % H = d.hand(:,126:135); %pretty irrelevant as we only have 1 hand


    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Normalize maxF & AUC by each subject's MVC (per hand)
    %MV we can only do this for last 10 trials
    for k=1:10  %(for each trial)
        if ~isnan(maxF(i,k)) % if they had to squeeze
            
                maxFNorm(i,k) = maxF(i,k)./d.MVC(i,k,1);
                AUCNorm(i,k) = AUC_R(i,k)./d.MVC(i,k,1);
          
        elseif isnan(maxF(i,k))
            maxFNorm(i,k) = nan;
            AUCNorm(i,k) = nan;
        end
    end
    for effort = 1:5 % get the frequency "map"
        for reward = 1:5
            filter = (d.stake (i,1:125) == ss(reward)) ...
                & (d.effort(i,1:125) == es(effort));
            filter2 = (d.stake (i,126:135) == ss(reward)) ...
                & (d.effort(i,126:135) == es(effort));
            
            freqmap(effort,reward,i) = nanmean( y(filter) );
            if sum(filter)==5
                choicemap(effort, reward, i, :) = y(filter);
                decisiont(effort, reward, i, :) = dt{subs}(i, filter);
                yesMap(effort,reward,i,:) = yesLoc(filter);
            
            else
                % there are not 5 trials for this condition for this subject
                warning('bad data sub %g eff %g sta %g has %g', i, effort, reward, sum(filter));
                if sum(filter)==4  % if there were only 4 matching trials,
                    choicemap(effort, reward, i, :) = [nan; y(filter)]; % 4: add a nan for block 1
                    decisiont(effort, reward, i, :) = [nan; dt{subs}(i, filter)'];
                    
                    
                    
                    
                end
            end
        end
    end
    end
    % And now remove all trials where decision time was less than certain
    % value (e.g. 400ms) - replace them with nan
    for effort = 1:5
        for reward = 1:5
            for block = 1:5
                if decisiont(effort,reward,i,block) <= 0.4
                    choicemap(effort,reward,i,block) = nan;
                    maxF(effort,reward,i,block) = nan;
                    decisiont(effort,reward,i,block) = nan;
                    yesMap(effort,reward,i,block) = nan;
                end
            end
        end
    end

grpD.freqmap{subs}=freqmap; % Note this has not had erroneous trials removed %%
grpD.choicemap{subs}=choicemap; % ALL_CHOICEMAP { subs } ( EFFORT, REWARD, SUBJECT, BLOCK )
grpD.decisiont{subs}=decisiont;
grpD.maxF{subs}=maxF;
maxFData{subs}=maxFNorm;
grpD.yesMap{subs}=yesMap;

choices = nanmean(grpD.choicemap{1},4);

D.Yestrial=D.Yestrial(:,1:125);
D.Yestrial(isnan(D.Yestrial))=0;
D.stake = D.stake(:,1:125);
D.effort = D.effort(:,1:125);

save('AGT_grpD_CHAPAS40')

%% Quick quality check. 
% lets have a look at the general trend of the data across reward and
% effort levels. 
%first  for the whole group? I think it may be best to look at reward and
%effort seperately. 

c = @cmu.colors;

close all
grp_rew_2D = squeeze(mean(freqmap,1))';
H1=shadedErrorBar(1:5,nanmean(grp_rew_2D),   ...
    nanstd(grp_rew_2D./sqrt(36)),'lineprops',...
    {'color', c('royal purple')},...
    'patchSaturation',0.3);
hold on 
grp_eff_2D = squeeze(mean(freqmap,2))';
H2=shadedErrorBar(1:5,nanmean(grp_eff_2D),   ...
    nanstd(grp_eff_2D./sqrt(36)),'lineprops',...
    {'color',c('air force blue')},  ...
    'patchSaturation',0.3);
axis square
ylim([0 1.1]);xlim([0 6])
ax=gca;
set(ax,'fontWeight','bold','fontSize',16,'XTick',[1:1:5], ...
  'XTickLabel',{'10','38','52','66','80'})
xlabel('Effort (MVC%)')
ylabel('Prop. Offers Accepted')
ylim([0.2 1])
title('Effort deters motivated behaviour');
% hold off


[lgd, icons, plots, txt] = legend([H2.mainLine],{'Effort'});



subjects = [1:34];
close all
% now for each individual. 
% create matrices compatible with the errorbar function in order to
% generate errorbars for each individual's plots. 
% first rearrange the choice map so that we can have each subjects choice
% data individually available to create error bars for each. 
permute_choicemap_rew = squeeze(mean(permute(choicemap,[1,2,4,3]),1));
permute_choicemap_eff = squeeze(mean(permute(choicemap,[1,2,4,3]),2));
dat_rew = permute(permute_choicemap_rew,[2,1,3]);
dat_eff = permute(permute_choicemap_eff,[2,1,3]);

% now generate a subplot for each individual with their performance. 
%create legend indices
close all
figure()

subj = size(D.R,1);
exclude = [26:28 32:33];
for i = 1:size(D.R,1)
  subplot(7,7,i)
  hold on
  H3= shadedErrorBar(1:5,nanmean(dat_rew(:,:,i),1), ...
    nanstd(dat_rew(:,:,i)./sqrt(5)),'lineprops',...
    {'color', c('royal purple')},...
    'patchSaturation',0.3);
  hold on
  H4= shadedErrorBar(1:5,nanmean(dat_eff(:,:,i),1), ...
    nanstd(dat_eff(:,:,i)./sqrt(5)),'lineprops',...
    {'color', c('air force blue')},...
    'patchSaturation',0.3);
  hold off
  hold off
  axis square
  ylim([0 1.1]); xlim([0 6]);
  title(['Subject ' num2str(subjects(i))]);
  %ax=gca;
  %set(ax,'fontWeight','bold','fontSize',12,'XTick',[1:1:5], ...
   % 'XTickLabel',{'1','2','3','4','5'})
end

[lgd, icons, plots, txt] = legend([H3.mainLine H4.mainLine],...
  {'Reward','Effort'});

 [ax1,h1]=suplabel('Effort/Reward Level');
 [ax2,h2]=suplabel('Prop. Accepted','y');
 
 
% have a closer look at those not engaging with the task well. 
close all
clear i
figure()
subjects = [1:33];
exclude = [1 26:28 32:33];
subNames = {'Model Subject','Subject '}
for i = 1:size(exclude,2)
  subplot(ceil(sqrt(length(exclude))),ceil(sqrt(length(exclude)))...
    ,i)
  hold on
  H3= shadedErrorBar(1:5,nanmean(dat_rew(:,:,exclude(i)),1), ...
    nanstd(dat_rew(:,:,i)./sqrt(5)),'lineprops',...
    {'color', c('royal purple')},...
    'patchSaturation',0.3);
  hold on
  H4= shadedErrorBar(1:5,nanmean(dat_eff(:,:,exclude(i)),1), ...
    nanstd(dat_eff(:,:,i)./sqrt(5)),'lineprops',...
    {'color', c('air force blue')},...
    'patchSaturation',0.3);
  hold off
  hold off
  axis square
  ylim([0 1.1]); xlim([0 6]);
  if i ==1
     title(subNames{1});
  else
  title([subNames{2} num2str(subjects(exclude(i)))]);
  %ax=gca;
  %set(ax,'fontWeight','bold','fontSize',12,'XTick',[1:1:5], ...
   % 'XTickLabel',{'1','2','3','4','5'})
  end
end

[lgd, icons, plots, txt] = legend([H3.mainLine H4.mainLine],...
  {'Reward','Effort'});

 [ax1,h1]=suplabel('Effort/Reward Level');
 [ax2,h2]=suplabel('Prop. Accepted','y');
 





% how about in three dimensions? 

% first create a three dimensional vector with subjects on the z axis
Permute_3D = permute(freqmap,[1,2,3]);
figure()
for i = 1:size(D.R,1)
    subplot(7,7,i)
    hold on 
    surf(Permute_3D(:,:,i));
    hold off 
    colormap(jet);
    view(50,30);
end 
    colorbar

 
% close figures
close
% dock all figures 
set(0,'DefaultFigureWindowStyle','docked') 
% plot acceptance vs apathy rating wth cut off visualised. 
figure()
scatterRegress(Questionnaires.AMI_T,accept);
hold on
plot(ones(1,120)'*1.77,1:120,'--b','LineWidth',2);



% ven diagram with cut offs 
% my cut offs are: 1.91 for AMI, 1 std for Calgary and for PANS. 
% find how many have all three 

CALG_Cut = nanmean(Questionnaires.CALGARY_TOTAL)+nanstd(Questionnaires.CALGARY_TOTAL);
AP_Cut_healthy = 1.91;
AP_Cut_Schizophrenia = 2.25;
PANS_Cut = nanmean(Questionnaires.PANS_NEG_TOTAL)+nanstd(Questionnaires.PANS_NEG_TOTAL);
BNSS_Cut = nanmean(Questionnaires.BNSS_TOTAL)+nanstd(Questionnaires.BNSS_TOTAL);
AP_std=(find(Questionnaires.AMI_T>=nanmean(Questionnaires.AMI_T)+nanstd(Questionnaires.AMI_T)));
AP_dep_neg = (find(Questionnaires.CALGARY_TOTAL>CALG_Cut & ...
    Questionnaires.AMI_T>=1.91 & ...
    Questionnaires.BNSS_TOTAL>=BNSS_Cut));

AP_dep = size(find(Questionnaires.CALGARY_TOTAL>=CALG_Cut & ...
    Questionnaires.AMI_T>=AP_Cut_Schizophrenia));

AP = size(find(Questionnaires.AMI_T>=1.91));

P_neg = size(find(Questionnaires.PANS_NEG_TOTAL>=PANS_Cut));
B_neg = size(find(Questionnaires.BNSS_TOTAL>=BNSS_Cut));
AP_neg = size(find(Questionnaires.AMI_T>=AP_Cut_Schizophrenia & ...
    Questionnaires.PANS_NEG_TOTAL>=PANS_Cut));
Dep = size(find(Questionnaires.CALGARY_TOTAL>=CALG_Cut));
Dep_neg = size(find(Questionnaires.CALGARY_TOTAL>=CALG_Cut & ...
    Questionnaires.PANS_NEG_TOTAL>=PANS_Cut));

% create vector with appropriate numbers 
%Z is a 7 element vector [z1 z2 z3 z12 z13 z23 z123]
close all;
c = @cmu.colors;
Z = [15 9 20 0 9 7 3];
Ven=venn(Z,...
    'FaceColor',{c('coral'),c('royal purple'),c('pastel purple')},...
    'FaceAlpha',{0.9,0.9,0.7});
hold on 
title('Venn Diagram for Apathy, Depression and Negative syndrome',...
    'FontSize',18)























































