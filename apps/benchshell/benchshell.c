#include <stdio.h>

#define PROMPT ">"
#define MAX_INPUT 1024
#define MAX_CALL 2

typedef struct {
	char * call;
	char * desc;
	void (*func)(void *);
} cmd_t;

const cmd_t CMD_NOT_FOUND = { .call = NULL, .desc = NULL, .func = NULL};
static int nCommands = 0;
static cmd_t * cmds;


void registerCommand(cmd_t cmd)
{
	nCommands++;
	cmds = (cmd_t *)realloc(cmds,nCommands*sizeof(cmd_t));
	cmds[nCommands-1] = cmd;

}

int commandNotFound(cmd_t cmd) 
{
	return (cmd.call == CMD_NOT_FOUND.desc && 
				cmd.desc == CMD_NOT_FOUND.desc && 
				cmd.func == CMD_NOT_FOUND.func);
}


void listAvailableCommands(void *arg)
{
	int i,c;
	for (i=0; i<nCommands; i++) {
			printf("%*s: %s\n",MAX_CALL,cmds[i].call,cmds[i].desc);
	}
}

cmd_t findMatchingCommand(char *input)
{
	int i,c;
	for (i=0; i<nCommands; i++) {
		if (!strcmp(input,cmds[i].call))
			return cmds[i];
	}
	return CMD_NOT_FOUND;
}

int main(int argc, char ** argv)
{
	int i;

	cmd_t cmd_listCommands;
	cmd_listCommands.call = "l";
	cmd_listCommands.desc = "List all available commands";
	cmd_listCommands.func = &listAvailableCommands;
	registerCommand(cmd_listCommands);
		
	listAvailableCommands(NULL);

	while (1)
	{
		printf("%s ",PROMPT);
		char *input = (char *)malloc(MAX_INPUT);
		input = fgets(input,MAX_INPUT,stdin);

		if (!input)
			break;

		// remove newline
		char *p = strchr(input,'\n');
		if (p)
			*p = '\0';

		cmd_t cmd = findMatchingCommand(input);
		
		if (commandNotFound(cmd)) {
			printf("Command not found.\n");
			continue;
		}
		
		cmd.func(NULL);





	}
	printf("\n");



}
