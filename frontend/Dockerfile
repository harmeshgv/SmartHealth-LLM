# Start with a compatible Node.js base image
FROM node:25-alpine
WORKDIR /app
RUN npm install -g npm@11.6.2
COPY . .
RUN npm install
EXPOSE 3000
CMD [ "npm", "start" ]
