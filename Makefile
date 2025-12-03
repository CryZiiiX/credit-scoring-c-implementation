CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -D_POSIX_C_SOURCE=200809L
LDFLAGS = -lm

SRC_DIR = src
BUILD_DIR = build
TARGET = $(BUILD_DIR)/credit_risk_predictor

# Source files
SRCS = $(SRC_DIR)/main.c \
       $(SRC_DIR)/utils/utils.c \
       $(SRC_DIR)/utils/memory_manager.c \
       $(SRC_DIR)/utils/csv_parser.c \
       $(SRC_DIR)/data/data_loader.c \
       $(SRC_DIR)/data/data_splitter.c \
       $(SRC_DIR)/preprocessing/preprocessing.c \
       $(SRC_DIR)/preprocessing/scaler.c \
       $(SRC_DIR)/preprocessing/encoder.c \
       $(SRC_DIR)/models/logistic_regression.c \
       $(SRC_DIR)/models/decision_tree.c \
       $(SRC_DIR)/evaluation/metrics.c \
       $(SRC_DIR)/evaluation/confusion_matrix.c

OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/*/*.o $(TARGET)
	@echo "Clean complete"

run: $(TARGET)
	./$(TARGET)
