#pragma once
#include "../plhe/params.h"
#include "../channel/channel.h"

typedef struct client_s client_t;

client_t *client_create(void);
void      client_destroy(client_t *client);

void client_query(client_t *client, channel_t *channel,
                  const int *sub_rows, const int *col_indx);

void client_recover(client_t *client, channel_t *channel,
                    uint8_t results[B_CHUNKS]);
