# Built
# transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # # transforms.Normalize(mean=mean, std=std)
# ])

# # Without transforms
# dataset = CustomDataset(csv_path=config.CSV_FILE_PATH)
#
# fig = plt.figure()
#
# for i in range(4):
#     sample = dataset[i]
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title(sample["label"])
#     ax.axis('off')
#     print(sample["image"].size)
#     plt.pause(0.001)
#
# plt.show()
#
# # With transforms
# dataset = CustomDataset(csv_path=config.CSV_FILE_PATH, transform=transform)
#
# fig = plt.figure()
#
# for i in range(4):
#     sample = dataset[i]
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title(sample["label"])
#     ax.axis('off')
#     print(sample["image"].shape)
#     plt.imshow(torch.movedim(sample["image"], 0, 2))
#     plt.pause(0.001)
#
# plt.show()
