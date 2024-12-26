NUM_CLASSES = 2
CLASS_NAMES = [ 'BG', 'Crack']
perturb_steps=5
epsilon=0.102 
step_size = 0.051 / 4
loss_fn="cent"
category="Madry"

def FGA_PGD(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    Kappa = torch.zeros(len(data))
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        #x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial feature
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        #x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    return Kappa


  rank  = FGA_PGD(model.fc,fea_input,targets, epsilon,step_size,perturb_steps,loss_fn="cent",category="Madry",rand_init=True)
  print("rank:",rank)
  rank = rank.cuda()  

  pred = F.softmax(outputs, dim=1)
  confidence,_ = torch.max(pred, dim=1)

  GA_List = []
  for ind in range(perturb_steps + 1):
      GA_Cur = torch.where(rank == ind, torch.ones_like(rank), torch.zeros_like(rank))
      #print(GA_Cur)
      if GA_Cur.sum() >0:
          expected_GA = (GA_Cur * confidence).sum()/GA_Cur.sum()
          GA_List.append(expected_GA)     

  loss_aicl = 0
  for ind in range(len(GA_List)-1):
      if GA_List[ind]> GA_List[ind+1]:
          loss_aicl = loss_aicl + criterionL1(GA_List[ind],GA_List[ind+1].detach().clone())  #.detach().clone()

  print("loss_aicl",loss_aicl)
