import torch

DEVICE_TYPE = "cuda"

if True:
    print(f"{torch.cuda.is_available()=}")
    t = torch.empty(size=(2, 3), dtype=torch.float, device="cuda")
    ta = torch.abs(t)
    print(ta)


if False:
    stream = torch.cuda.Stream()
    torch.cuda.empty_cache()
    # Create some tensors
    x = torch.empty(1000, 1000, device=DEVICE_TYPE)
    y = torch.empty(1000, 1000, device=DEVICE_TYPE)

    # ----------------- DOJO Graph -----------------
    print("\nDOJO Graph execution:")

    with torch.cuda.stream(stream):
        # Step 1: Create a DOJO graph
        graph = torch.cuda.CUDAGraph()

        graph.capture_begin()
        z = x + y
        z = z**2
        z = torch.relu(z)
        graph.capture_end()

print("Done.")
