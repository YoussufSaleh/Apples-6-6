

%% Pipeline for analysis of the AGT which requires effortful decisions only
% Created by YS on 6/8/2018. 

%useful comments below by CLH from parent script regarding exclusions. It
%looks like at least 15 patients will be excluded either because they do NOT have ...
% SVD or because 
% Excluded OXVASC patients: 5 6 13 21 24 30 31 48 (VD Didn;t get task
% (check this) 53 (didn't engage with task - "easier to say yes than to
% think") plus no SVD seen for: 3 6 9:14 24 28 29
%17/9 - change script to exclude any trials that were "mistakes" from both
%logistic regression, and from choicemap (replace with nan), and then
%nanmean for each subject.
% 11/9/15 - adapt to load all data into a single matrix


%create cell array allStudyGroups which contains cell arrays with the title and
%data sets for each individual in the relevant cohort. 

%clear 

allStudyGroups = { ...
    % subGroup FILE TEMPLATE,  SUBJECT NUMBERS
    
    {'Results_YS_ApplesSVD_%02d', [1:2 4 7 8 15:20 22:23 25:27 32:47 49:52 54:64 66:67 69 71:81 82 83:98 100:104]}};% data to be formatted into file format
groupName = {'SVD'};
code = [1:2 4 7 8 15:20 22:23 25:27 32:47 49:52 54:64 66:67 69 71:81 82 83:98 100:104];
for subGroup=1:length(allStudyGroups)
    alldata=[]; % blank array
    for i = 1:length(allStudyGroups{subGroup}{2}) % go through subjects in subGroup

        data=load(sprintf(allStudyGroups{subGroup}{1},code(i)));
        d=data.result.data;  % retrieve trial data
        samp{subGroup}{i}{1}=[]; samp{subGroup}{i}{2}=[];
        for j = 1:length(allStudyGroups{subGroup}{2}) % for each subject
        for t = 1:216 % for each trial
            AUC1{subGroup}(j,t) = nansum(d(t).data1); % extract AUC data before ...
            AUC2{subGroup}(j,t) = nansum(d(t).data2); % removing data1 and 2 fields
          %  if 0 % save squeeze data?
          %      f1 =  d(t).data1;
                %f2 =  d(t).data2;
          %      samp{subGroup}{i} = nancat(2, samp{subGroup}{i}, f1+f2);
          %  end
        end
        end

        d=rmfield(d, 'data1'); % don't process the actual squeeze data
        d=rmfield(d, 'data2');
        if isempty(alldata) % first subject: new structure
          alldata = d;
        else % subsequent subjects - make sure fields match
            d=ensureStructsAssignable(d, alldata);
            alldata=[alldata;d]; % and then add a new row
        end
    end
    d=transpIndex(alldata);
    alld{subGroup}=d;
 
end
%save all_groups_data alld allStudyGroups

%% now want to combine all of above into a single array for easier processing...
% Files are all labeled with numbers, and will be added to so to hard to do
% it this way,

% variables = {'hand' 'block' 'trialIndex' 'R' 'starttrial' 'effort' 'stake'...
%      'startStim' 'startChoice' 'endChoice' 'endNotrial' 'totalReward' 'startresponse' ...
%      'endresponse' 'Yestrial' 'maximumForce'};
%
% for k = 1:length(variables); %for each variable
hand = [];
block = [];
trialIndex = [];
R = [];
starttrial = [];
effort = [];
stake = [];
startStim = [];
startChoice = [];
endChoice = [];
endNotrial = [];
totalReward = [];
startresponse = [];
endresponse = [];
Yestrial = [];
reward = [];
maximumForce = [];
AUCA = [];
AUCB = [];
MVC1 = [];
MVC2 = [];
vigour = [];

for j=1:length(allStudyGroups); %for each subGroup
    hand = [hand;alld{j}.hand];
    block= [block;alld{j}.block];
    trialIndex = [trialIndex;alld{j}.trialIndex];
    R = [R;alld{j}.R];
    starttrial = [starttrial;alld{j}.starttrial];
    effort = [effort;alld{j}.effort];
    stake = [stake;alld{j}.stake];
    startStim = [startStim;alld{j}.startStim];
    startChoice = [startChoice;alld{j}.startChoice];
    endChoice = [endChoice;alld{j}.endChoice];
    endNotrial = [endNotrial;alld{j}.endNotrial];
    totalReward = [totalReward;alld{j}.totalReward];
    startresponse = [startresponse;alld{j}.startresponse];
    endresponse = [endresponse;alld{j}.endresponse];
    Yestrial = [Yestrial;alld{j}.Yestrial];
    reward = [reward;alld{j}.reward];
    maximumForce = [maximumForce;alld{j}.maximumForce];
    vigour = [vigour;alld{j}.maximumForce-alld{j}.effort];
    MVC1 = [MVC1;alld{j}.MVC(:,:,1)];
    MVC2 = [MVC2;alld{j}.MVC(:,:,2)];
    AUCA = [AUCA;AUC1{j}];
    AUCB = [AUCB;AUC2{j}];
    
end
D.hand = hand;
D.block = block;
D.trialIndex = trialIndex;
D.R = R;
D.starttrial = starttrial;
D.effort = effort;
D.stake = stake;
D.startStim = startStim;
D.startChoice = startChoice;
D.endChoice = endChoice;
D.endNotrial = endNotrial;
D.totalReward = totalReward;
D.startresponse = startresponse;
D.endresponse = endresponse;
D.Yestrial = Yestrial;
D.reward = reward;
D.maximumForce = maximumForce;
D.MVC1 = nanmean(MVC1')';
D.MVC2 = nanmean(MVC2')';
D.AUCA = AUCA;
D.AUCB = AUCB;
D.vigour = vigour;
clear  hand block trialIndex R starttrial effort stake startStim startChoice endChoice endNotrial totalReward startresponse endresponse Yestrial reward maximumForce AUCA AUCB MVC1 MVC2

 
%% 
    

%% now try and model it
HEATMAP = 0;
subGroup = 1;
d=D; % allows reading from above script...
SAVE_FORCE = false;

clear  y max Fb dev stats freqmap choicemap decisiont maxforce hand MVC forceD1 forceD2 st numExcl
clf
allDat=[];
for i=1:size(d.R,1) % for each subject
    es =  [0.1 0.24 0.38 0.52 0.66 0.8]; %(unique(d.effort(i,:)));
    eff = [0.1 0.24 0.38 0.52 0.66 0.8]; %(unique(d.effort(i,:)));
    ss =  [1 3 6 9 12 15]'; %(unique(d.stake(i,:)))';
    stk = [1 3 6 9 12 15]';%(unique(d.stake(i,:)))';
    y = d.Yestrial(i,:)'; % did they accept?
    y = y(37:end); %Remove the first 36 trials before proceeding further
    y(isnan(y))=0; % compensate for someone's incompetence
    dt = d.endChoice-d.startChoice;
    dt = dt(:,37:end); %exclude practice session
    maxF = d.maximumForce(:,37:end); %exclude practise session
%     AUC1 = d.AUCA(:,37:end);
%     AUC2 = d.AUCB(:,37:end);
    H = d.hand(:,37:end);
    %Normalize maxF by each subject's MVC (per hand)
    for k=1:180 % (for each trial)
        if ~isnan(maxF(i,k))
            if H(i,k) == 1
                maxFNorm(i,k) = maxF(i,k)./d.MVC1(i);
            elseif H(i,k) ==2
                maxFNorm(i,k) = maxF(i,k)./d.MVC2(i);
            end
        elseif isnan(maxF(i,k))
            maxFNorm(i,k) = nan;
        end
    end
    for effort = 1:6 % get the frequency "map"
        for reward = 1:6
            filter = ... % select trials of this condition
                  (d.stake (i,37:end) == ss(reward)) ...
                & (d.effort(i,37:end) == es(effort));
            freqmap(effort,reward,i) = nanmean( y(filter) );    
            if sum(filter)==5  % frequency map including block!
                % choicemap ( effort, reward, subject, block )
                choicemap(effort, reward, i, :) = y(filter);
                decisiont(effort, reward, i, :) = dt(i, filter);
                maxforce( effort, reward, i, :) = maxFNorm(i, filter);
                maxVig( effort,reward,i,:) = 
                hand(     effort, reward, i, :) = H(i, filter);
%                 AUCA(     effort, reward, i, :) = AUC1(i,filter);
%                 AUCB(     effort, reward, i, :) = AUC2(i,filter);
                if SAVE_FORCE
                    forceD1(     effort, reward, i, :) = AUC1{subGroup}(i,filter);
                    forceD2(     effort, reward, i, :) = AUC2{subGroup}(i,filter);
                end
            else
                % there are not 5 trials for this condition for this subject
                warning('bad data sub %g eff %g sta %g has %g', i, effort, reward, sum(filter));
                if sum(filter)==4  % if there were only 4 matching trials,
                    choicemap(effort, reward, i, :) = [nan; y(filter)]; % 4: add a nan for block 1
                    decisiont(effort, reward, i, :) = [nan; dt(i, filter)'];
                    maxforce( effort, reward, i, :) = [nan; maxFNorm(i,filter)'];
                    hand(     effort, reward, i, :) = [nan; H(i,filter)'];
%                     AUCA(     effort, reward, i, :) = AUC1(i,filter);
%                     AUCB(     effort, reward, i, :) = AUC2(i,filter);
                    if SAVE_FORCE
                      forceD1(  effort, reward, i, :) = [nan; AUC1{subGroup}(i,filter)'];
                      forceD2(  effort, reward, i, :) = [nan; AUC2{subGroup}(i,filter)'];
                    end
                elseif sum(filter)==6 % if there were 6 rather than 5 matching trials
                    f2=find(filter); f2=f2(2:end); % remove the first matching trial from the filter
                    choicemap(effort,reward,i,:) = y(f2);
                    decisiont(effort,reward,i,:) = dt(i,f2);
                    maxforce( effort,reward,i,:) = maxFNorm(i,f2);
                    hand(     effort,reward,i,:) = H(i,f2);
%                     AUCA(     effort, reward, i, :) = AUC1(i,filter);
%                     AUCB(     effort, reward, i, :) = AUC2(i,filter);
                    if SAVE_FORCE
                        forceD1(  effort,reward,i,:) = AUC1{subGroup}(i,f2);
                        forceD2(  effort,reward,i,:) = AUC2{subGroup}(i,f2);
                    end
                end
            end
        end
    end
    % And now remove all trials where decision time was less than certain
    % value (e.g. 400ms) - replace them with nan
    for effort = 1:6
        for reward = 1:6
            for block = 1:5
               if decisiont(effort,reward,i,block) <= 0.4
                   choicemap(effort,reward,i,block) = nan;
                   maxforce(effort,reward,i,block) = nan;
                   hand(effort,reward,i,block) = nan;
                   decisiont(effort,reward,i,block) = nan;
%                    AUCA(effort,reward,i,block) = nan;
%                    AUCB(effort,reward,i,block) = nan;
               end
            end
        end
    end
    
    if SAVE_FORCE
        MVC(i,1) = nanmean(unique(alld{subGroup}.MVC(i,1:6,1))); % Calculate MVC for each hand
        MVC(i,2) = nanmean(unique(alld{subGroup}.MVC(i,1:6,2))); % Think that 1 = right
    end
    
    grpD.freqmap{subGroup}=freqmap;
    grpD.choicemap{subGroup}=choicemap; % ALL_CHOICEMAP { subGroup } ( EFFORT, REWARD, SUBJECT, BLOCK )
    grpD.decisiont{subGroup}=decisiont;
    grpD.maxforce{subGroup}=maxforce;
    grpD.MVC1 = D.MVC1;
    grpD.MVC2 = D.MVC2;
%     grpD.AUCA{subGroup} = AUCA;
%     grpD.AUCB{subGroup} = AUCB;
    if SAVE_FORCE
        grpD.MVC{subGroup} = MVC;
        grpD.forceD1{subGroup} = forceD1;
        grpD.forceD2{subGroup} = forceD2;
    end
    grpD.hand{subGroup} = hand;
end
D.maxFnorm=maxFNorm;

save('AGT_SVD_Final');

