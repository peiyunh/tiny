%  FILE:   cnn_setup_imdb.m
%
%    This function reads WIDER FACE dataset and generates a MATLAB struct
%    variable that is more friendly and eaiser to work with. 
%  
%  INPUT:  
% 
%  OUTPUT: imdb (a more user-friendly struct variable for the dataset)

function imdb = cnn_setup_imdb(varargin)

opts.dataDir = fullfile('data','widerface') ;
opts = vl_argparse(opts, varargin) ;

% construct imdb
imdb = struct();
imdb.imageDir = opts.dataDir;
imdb.images = struct();
imdb.labels = struct();

imdb.images.name = {};

cnt = 0;
% train
load(fullfile(opts.dataDir, 'wider_face_train.mat'));
for i = 1:numel(event_list)
    imageDir = fullfile('WIDER_train/images', event_list{i});
    imageList = file_list{i};
    bboxList = face_bbx_list{i};
    for j = 1:numel(imageList)
        cnt = cnt + 1;
        imagePath = fullfile(imageDir, [imageList{j} '.jpg']);
        imdb.images.name{cnt} = imagePath;
        
        info = imfinfo(fullfile(opts.dataDir, imagePath));
        imdb.images.size(cnt,1:2) = [info.Height info.Width]; 
        
        imdb.images.set(cnt) = 1;
        rects = bboxList{j};
        imdb.labels.rects{cnt} = horzcat(...
            rects(:,[1 2]), rects(:,[1 2])+rects(:,[3 4])-1);
        imdb.labels.eventid(cnt) = i;
    end
end

% setup event list ( index is consistent with face_bbx_list )
%load(fullfile(opts.dataDir, 'event_diffmap.mat'));
%imdb.events.name = event_list; 
%for i = 1:numel(event_list)
%    imdb.events.diff(i) = diffmap(event_list{i}); 
%end

% clear variables
fprintf('Setup imdb: processed %d images.\n', cnt);
clear face_bbx_list event_list file_list;

% val
load(fullfile(opts.dataDir, 'wider_face_val.mat'));
for i = 1:numel(event_list)
    imageDir = fullfile('WIDER_val/images', event_list{i});
    imageList = file_list{i};
    bboxList = face_bbx_list{i};
    for j = 1:numel(imageList)
        cnt = cnt + 1;
        imagePath = fullfile(imageDir, [imageList{j} '.jpg']);
        imdb.images.name{cnt} = imagePath;
        
        info = imfinfo(fullfile(opts.dataDir, imagePath));
        imdb.images.size(cnt,1:2) = [info.Height info.Width]; 
        
        imdb.images.set(cnt) = 2;
        rects = bboxList{j};
        imdb.labels.rects{cnt} = horzcat(...
            rects(:,[1 2]), rects(:,[1 2])+rects(:,[3 4])-1);
        imdb.labels.eventid(cnt) = i;
    end
end
fprintf('Setup imdb: processed %d images.\n', cnt);
clear face_bbx_list event_list file_list;

% test
load(fullfile(opts.dataDir, 'wider_face_test.mat'));
for i = 1:numel(event_list)
    imageDir = fullfile('WIDER_test/images', event_list{i});
    imageList = file_list{i};
    for j = 1:numel(imageList)
        cnt = cnt + 1;
        imagePath = fullfile(imageDir, [imageList{j} '.jpg']);
        imdb.images.name{cnt} = imagePath;
        
        info = imfinfo(fullfile(opts.dataDir, imagePath));
        imdb.images.size(cnt,1:2) = [info.Height info.Width]; 
        
        imdb.images.set(cnt) = 3;
        imdb.labels.eventid(cnt) = i;
    end
end
fprintf('Setup imdb: processed %d images.\n', cnt);
clear event_list file_list;
