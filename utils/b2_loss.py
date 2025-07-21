import torch


def trans_b2_loss(pred, gt, mask):
    dist = [1, 3, 9, 27]

    T = pred.shape[1]
    all_loss = []
    indiv_l2_loss = torch.nn.L1Loss(reduction="none")(pred, gt)

    for n in dist:
        sigma_sums = []
        for i in range(0, (T // n)):
            masked_indiv_l2_loss = indiv_l2_loss[:, i * n : i * n + n][
                mask[:, i * n + n - 1] == 1
            ]
            if len(masked_indiv_l2_loss) == 0:
                continue
            masked_indiv_l2_loss = torch.sum(masked_indiv_l2_loss, dim=1)

            rms_norm = torch.sqrt(torch.mean(masked_indiv_l2_loss**2, dim=1))
            sigma_sums.append(rms_norm)

        if len(sigma_sums) > 0:
            all_loss.append(torch.mean(torch.cat(sigma_sums, dim=0)))

    if len(all_loss) > 0:
        total_loss = torch.sum(torch.stack(all_loss))
        return total_loss
    else:
        return torch.tensor(0.0).to(pred.device)
