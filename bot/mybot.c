#define _DEFAULT_SOURCE
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>

#include "botlib.h"

/* For each bot command, private message or group message (but this only works
 * if the bot is set as group admin), this function is called in a new thread,
 * with its private state, sqlite database handle and so forth.
 *
 * For group messages, this function is ONLY called if one of the patterns
 * specified as "triggers" in startBot() matched the message. Otherwise we
 * would spawn threads too often :) */
void handleRequest(sqlite3 *dbhandle, BotRequest *br) {
    int64_t set_target = 0;

    sds targetstr = kvGet(dbhandle,"target");
    if (targetstr) {
        set_target = strtoll(targetstr,NULL,0);
        sdsfree(targetstr);
    }

    if (br->argc == 1 && !strcasecmp(br->argv[0],"!target")) {
        if (set_target) {
            botSendMessage(br->target,"A target is already set. Use !forget",br->msg_id);
            return;
        }

        char buf[64];
        snprintf(buf,sizeof(buf),"%lld",(long long)br->target);
        kvSet(dbhandle,"target",buf,0);

        /* We also need to remember who set the target, so we will
         * accept !forget commands only from they. */
        snprintf(buf,sizeof(buf),"%lld",(long long)br->from);
        kvSet(dbhandle,"owner",buf,0);
        botSendMessage(br->target,"Thanks, I'll send failure images here.",br->msg_id);
    } else if (br->argc == 1 && !strcasecmp(br->argv[0],"!forget")) {
        sds ownerstr = kvGet(dbhandle,"owner");
        if (!ownerstr) {
            botSendMessage(br->target,"Target was not set yet.",br->msg_id);
            return;
        }
        int64_t owner = strtoll(ownerstr,NULL,0);
        sdsfree(ownerstr);

        if (owner != br->from) {
            botSendMessage(br->target,"I can only accept .",br->msg_id);
            return;
        }
        kvDel(dbhandle,"target");
        kvDel(dbhandle,"owner");
    }

    /* Accept only commands from set target. */
    if (!set_target || br->target != set_target) return;

    if (br->argc == 1 && !strcasecmp(br->argv[0],"!cam")) {
        FILE *fp = fopen("processed.jpg","r");
        if (fp == NULL) {
            botSendMessage(br->target,"No cam image available.",br->msg_id);
            return;
        }
        fclose(fp);
        botSendImage(br->target,"processed.jpg");
    }
}

// This is just called every 1 or 2 seconds. */
#define LOW_SCORE_COUNT_LIMIT 10
#define SCORE_THRESHOLD 0.01
void cron(sqlite3 *dbhandle) {
    static float sent_score = 0; // Confidence score of last sent image.
    int64_t target;
    
    /* Normally we sent each image that has a detection score larger than
     * the past. However the user at some point will change the print or
     * restart it: we need a way to reset. We take a counter of the amount of
     * times we see a low score, that is a score lower than the sent_score.
     * If the counter reaches the value of LOW_SCORE_COUNT_LIMIT, we reset
     * sent_score to zero again, assuming it is a new print.
     *
     * Why we do this? Because we don't want to flood the user with
     * images (especially on false positives), and sometimes even if the
     * print in progress is the same, the score naturally goes down again
     * because of different visual angle, printing head covering parts of
     * the print, and so forth. */
    static int low_score_count = 0;

    /* Get the stored target: the Telegram user/group ID we
     * will send images to. */
    sds target_str = kvGet(dbhandle,"target");
    if (target_str == NULL) return; // No configured target yet.
    target = strtoll(target_str,NULL,0);
    sdsfree(target_str);

    /* Get the current score from disk. The detector should update it. */
    FILE *fp = fopen("current_score.txt","r");
    if (fp == NULL) return; // No score file yet.

    char buf[64];
    fgets(buf,sizeof(buf),fp);
    float current_score = strtod(buf,NULL);
    fclose(fp);

    if (current_score > sent_score*1.15 && current_score > SCORE_THRESHOLD) {
        printf("Sending image with score: %.2f\n", current_score);
        botSendImage(target, "processed.jpg");
        sent_score = current_score;
        low_score_count = 0;
    } else if (current_score < sent_score && current_score < SCORE_THRESHOLD) {
        low_score_count++;
        if (low_score_count == LOW_SCORE_COUNT_LIMIT) {
            sent_score = 0;
            printf("Failed print removed from the bed? Reset sent_score to zero.\n");
        }
    }
}

int main(int argc, char **argv) {
    static char *triggers[] = {
        "!*",
        NULL,
    };
    startBot(TB_CREATE_KV_STORE, argc, argv, TB_FLAGS_NONE, handleRequest, cron, triggers);
    return 0; /* Never reached. */
}
