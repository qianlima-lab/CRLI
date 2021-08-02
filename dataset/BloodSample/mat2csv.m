load('TCK_data.mat')
mask_X = 1 - isnan(X);
mask_Xte = 1 - isnan(Xte);
X(isnan(X)) = 0;
Xte(isnan(Xte)) = 0;
cnt = 0;
for i = 1:707
    tmp = X(i,:,:);
    tmp = reshape(tmp,20,10);
    tmp = tmp';
    mat = reshape(tmp,1,20*10);
    mat = [Y(i),mat];
    if cnt == 0
            dlmwrite('BloodSample_TRAIN',mat,'delimiter',',');
    else
            dlmwrite('BloodSample_TRAIN',mat,'delimiter',',','-append');
    end
    cnt = cnt + 1;
end
cnt = 0;
for i = 1:176
    tmp = Xte(i,:,:);
    tmp = reshape(tmp,20,10);
    tmp = tmp';
    mat = reshape(tmp,1,20*10);
    mat = [Yte(i),mat];
    if cnt == 0
            dlmwrite('BloodSample_TEST',mat,'delimiter',',');
    else
            dlmwrite('BloodSample_TEST',mat,'delimiter',',','-append');
    end
    cnt = cnt + 1;
end
cnt = 0;
for i = 1:707
    tmp = mask_X(i,:,:);
    tmp = reshape(tmp,20,10);
    tmp = tmp';
    mat = reshape(tmp,1,20*10);
    mat = [1,mat];
    if cnt == 0
            dlmwrite('mask_BloodSample_TRAIN',mat,'delimiter',',');
    else
            dlmwrite('mask_BloodSample_TRAIN',mat,'delimiter',',','-append');
    end
    cnt = cnt +1;
end
cnt = 0;
for i = 1:176
    tmp = mask_Xte(i,:,:);
    tmp = reshape(tmp,20,10);
    tmp = tmp';
    mat = reshape(tmp,1,20*10);
    mat = [1,mat];
    if cnt == 0
            dlmwrite('mask_BloodSample_TEST',mat,'delimiter',',');
    else
            dlmwrite('mask_BloodSample_TEST',mat,'delimiter',',','-append');
    end
    cnt = cnt +1;
end

length = ones(1,707)*20 ;
length_mark = ones(707,20*10);
dlmwrite('length_BloodSample_TRAIN',length,'delimiter',',');
dlmwrite('lengthmark_BloodSample_TRAIN',length_mark,'delimiter',',')

length = ones(1,176)*20 ;
length_mark = ones(176,20*10);
dlmwrite('length_BloodSample_TEST',length,'delimiter',',');
dlmwrite('lengthmark_BloodSample_TEST',length_mark,'delimiter',',')

