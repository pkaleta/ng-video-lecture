import marimo

__generated_with = "0.9.31"
app = marimo.App(width="medium")


@app.cell
def __():
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    return F, nn, torch


@app.cell
def __(torch):
    torch.manual_seed(1337)
    B, T, C = 4, 8, 2
    x = torch.randn(B, T, C)
    x.shape
    return B, C, T, x


@app.cell
def __(B, C, T, torch, x):
    xbow = torch.zeros((B, T, C))
    for b in range(B):
        for t in range(T):
            xprev = x[b, :t+1]
            xbow[b, t] = torch.mean(xprev, axis=0)

    xbow
    return b, t, xbow, xprev


@app.cell
def __(torch):
    xx = torch.randn(3, 2)
    xx
    return (xx,)


@app.cell
def __(xx):
    xx.sum(axis=-1)
    return


@app.cell
def __(T, torch, x, xbow):
    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(1, keepdim=True)
    print(wei)
    print(x.shape)
    xbow2 = wei @ x  # (B, T, T) @ (B, T, C) --> (B, T, C)

    torch.allclose(xbow, xbow2)
    return wei, xbow2


@app.cell
def __(xbow):
    xbow[0]
    return


@app.cell
def __(torch):
    aa = torch.tril(torch.ones(3, 3))
    # print(aa.sum(1, keepdim=True))
    aa = aa / aa.sum(1, keepdim=True)
    bb = torch.randint(0, 10, (3, 2)).float()
    cc = aa @ bb
    print(aa)
    print(bb)
    print(cc)
    return aa, bb, cc


@app.cell
def __(torch):
    torch.set_printoptions(threshold=10000)
    return


@app.cell
def __(F, nn, torch):
    def ver4():
        B, T, C = 4, 8, 32
        x = torch.randn(B, T, C)

        head_size = 16
        query = nn.Linear(C, head_size, bias=False)
        key = nn.Linear(C, head_size, bias=False)
        q = query(x)  # (B, T, 16)
        k = key(x)    # (B, T, 16)
        wei = q @ k.transpose(-1, -2)  # (B, T, 16) @ (B, 16, T) --> (B, T, T)
        
        tril = torch.tril(torch.ones(T, T))
        # wei = torch.zeros((T, T))
        wei = wei.masked_fill(tril == 0, float("-inf"))
        # print(wei)
        wei = F.softmax(wei, dim=-1)
        print(wei[0])
        print(wei[1])
        return wei @ x

    ver4()
    return (ver4,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
