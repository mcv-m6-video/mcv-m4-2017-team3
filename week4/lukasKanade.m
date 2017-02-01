opticFlow = opticalFlowLK;
estimateFlow(opticFlow, img1);

OF = estimateFlow(opticFlow, img2);
flow_estimation_x = OF.Vx;
flow_estimation_y = OF.Vy;

flow_dx = shiftdim(OF(:,:,1))
flow_dy = shiftdim(OF(:,:,2))

GT = double(imread('../datasets_deliver2/gt/000157_10.png'));

gtX = (I(:,:,1)-2^15)/64;
gtY = (I(:,:,2)-2^15)/64;
gtMask = min(I(:,:,3),1);

gtX(GT_mask == 0) = 0;
gtY(GT_mask == 0) = 0;

GT(:,:,1) = gtX;
GT(:,:,2) = gtY;
GT(:,:,3) = gtMask;

errorx = gtX - flow_estimation_x;
errory = gtY - flow_estimation_y;
err = sqrt(errorx.*errorx+errory.*errory);
err(GT_mask==0) = 0;

msen = sum(err(:))/sum(err(:)>0)
pepn = length(find(err>3))/length(find(F_gt_val))
